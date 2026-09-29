#!/usr/bin/env python
# Copyright 2026 The NEST Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""SOC driver and reduced densities for spin-adapted NT-TDA."""

import numpy as np
from pyscf import lib
from pyscf.lib import logger

from nest.nttda.nttda import _sc_vector_slices
from nest.soc.soc import SOCBase, SpinFreeState, clebsch_gordan_rank1


class SOC(SOCBase):
    """Build SOC states from converged NTTDA ``deltaS=+1/0/-1`` objects."""

    _keys = {'tds', 'include_reference', 'orbitals'}

    def __init__(self, td0=None, *others, soctype='SOMF', include_reference=False):
        if td0 is None:
            if others:
                raise ValueError('The first NTTDA object cannot be None')
            tds = []
        elif isinstance(td0, (list, tuple)) and not others:
            tds = list(td0)
        else:
            tds = [td0, *others]

        super().__init__(soctype=soctype)
        self.tds = tds
        self.include_reference = include_reference
        self.orbitals = None

    def _initialize_states(self):
        if not isinstance(self.tds, (list, tuple)):
            raise TypeError('tds must be a list or tuple of NTTDA objects')
        if not self.tds:
            raise ValueError('Set tds to at least one NTTDA object before running SOC')

        mf = self._check_reference(self.tds)
        self._scf = mf
        if self.verbose is None:
            self.verbose = getattr(mf, 'verbose', logger.NOTE)
        if self.stdout is None:
            self.stdout = getattr(mf, 'stdout', None)

        log = logger.new_logger(self)
        log.info('\n')
        if not getattr(mf, 'converged', False):
            log.warn('Ground state SCF is not converged')
        for tdobj in self.tds:
            converged = getattr(tdobj, 'converged', None)
            if converged is None:
                continue
            unconverged = np.where(~np.asarray(converged, dtype=bool).reshape(-1))[0]
            if unconverged.size:
                log.warn(
                    'NTTDA deltaS=%s states %s are not converged',
                    tdobj.deltaS, unconverged.tolist(),
                )
        mo_occ = np.asarray(mf.mo_occ)
        if mo_occ.ndim != 1:
            raise ValueError('NTTDA SOC requires a spin-restricted ROKS/ROHF reference')
        cidx = np.where(mo_occ == 2)[0]
        oidx = np.where(mo_occ == 1)[0]
        vidx = np.where(mo_occ == 0)[0]
        self.orbitals = cidx, oidx, vidx, mo_occ.size
        states, has_delta0 = self._collect_states(self.tds, self.include_reference)
        if not has_delta0:
            if self.include_reference:
                logger.info(self, 'No deltaS=0 states are provided; reference state is included.')
            else:
                logger.info(self, 'No deltaS=0 states are provided; reference state is NOT included.')
                logger.info(self, 'To include the reference state, set include_reference=True before running SOC.')
        return states

    @staticmethod
    def _check_reference(tdobjs):
        mf = tdobjs[0]._scf
        mo_occ = np.asarray(mf.mo_occ)
        mo_coeff = np.asarray(mf.mo_coeff)
        if mo_coeff.ndim != 2:
            raise ValueError('NTTDA SOC requires a spin-restricted MO coefficient matrix')
        assert np.isrealobj(mo_coeff), 'NTTDA SOC requires real MO coefficients'
        for tdobj in tdobjs[1:]:
            assert np.isrealobj(tdobj._scf.mo_coeff), 'NTTDA SOC requires real MO coefficients'
            if not np.array_equal(np.asarray(tdobj._scf.mo_occ), mo_occ):
                raise ValueError('All NTTDA objects must share the same mo_occ')
            if not np.allclose(np.asarray(tdobj._scf.mo_coeff), mo_coeff, atol=1e-12, rtol=0):
                raise ValueError('All NTTDA objects must share the same mo_coeff')
        return mf

    def _collect_states(self, tdobjs, include_reference):
        cidx, oidx, vidx, _ = self.orbitals
        nc, no, nv = len(cidx), len(oidx), len(vidx)
        reference_spin = 0.5 * no
        states = []
        has_delta0 = False

        for tdobj in tdobjs:
            delta_s = int(getattr(tdobj, 'deltaS', getattr(tdobj, 'DeltaS', 99)))
            if delta_s not in (1, 0, -1):
                raise ValueError('deltaS should be -1, 0, or 1')
            if getattr(tdobj, 'e', None) is None or getattr(tdobj, 'xy', None) is None:
                raise ValueError('Run the NTTDA kernel before SOC')

            has_delta0 = has_delta0 or delta_s == 0
            spin = reference_spin + delta_s
            if spin < 0:
                raise ValueError('Invalid final spin inferred from deltaS')
            for root, energy in enumerate(tdobj.e):
                if delta_s == 1:
                    amplitude = self._unpack_deltap1(tdobj, root, nc, nv)
                elif delta_s == 0:
                    amplitude = self._unpack_delta0(tdobj, root, nc, no, nv)
                else:
                    amplitude = self._unpack_deltam1(tdobj, root, nc, no, nv)
                assert all(np.isrealobj(block) for block in amplitude), 'NTTDA SOC requires real amplitudes'
                states.append(SpinFreeState(
                    source=tdobj,
                    root=root,
                    energy=float(energy),
                    spin=float(spin),
                    amplitude=amplitude,
                    label=f'deltaS={delta_s} state {root + 1}',
                    delta_s=delta_s,
                ))

        if include_reference and not has_delta0:
            states.insert(0, SpinFreeState(
                source=None,
                root=None,
                energy=0.0,
                spin=float(reference_spin),
                amplitude=(
                    -1.0,
                    np.zeros((nc, no)),
                    np.zeros((nc, nv)),
                    np.zeros((no, nv)),
                    np.zeros((nc, nv)),
                ),
                label='reference',
                delta_s=0,
            ))
        return states, has_delta0

    @staticmethod
    def _unpack_deltap1(tdobj, root, nc, nv):
        return tdobj.xy[root][0].reshape(nc, nv)

    @staticmethod
    def _unpack_delta0(tdobj, root, nc, no, nv):
        slices = _sc_vector_slices(nc, no, nv)
        x = tdobj.xy[root][0].reshape(-1)
        return (
            x[slices['OO(1)']][0],
            x[slices['CO(1)']].reshape(nc, no),
            x[slices['CV(1)']].reshape(nc, nv),
            x[slices['OV(1)']].reshape(no, nv),
            x[slices['CV(0)']].reshape(nc, nv),
        )

    @staticmethod
    def _unpack_deltam1(tdobj, root, nc, no, nv):
        x = tdobj.xy[root][0].reshape(nc + no, no + nv)
        return x[:nc, :no], x[:nc, no:], x[nc:, :no], x[nc:, no:]

    @staticmethod
    def _add(gamma, rows, cols, block):
        gamma[np.ix_(rows, cols)] += block

    def _gamma_deltap1_deltap1(self, bra, ket):
        """Return ``<S+1,S+1|T^0_pq|S+1,S+1>`` in the MO basis."""
        cidx, _, vidx, nmo = self.orbitals
        gamma = np.zeros((nmo, nmo))
        b_cv = bra.amplitude
        k_cv = ket.amplitude
        factor = 1.0 / np.sqrt(2.0)

        # CV(ai)-CV(bj): p=j,q=i and p=a,q=b, respectively.
        self._add(gamma, cidx, cidx, factor * lib.einsum('ia,ja->ji', b_cv, k_cv))
        self._add(gamma, vidx, vidx, factor * lib.einsum('ia,ib->ab', b_cv, k_cv))
        return gamma

    def _gamma_deltap1_delta0(self, bra, ket):
        """Return ``<S+1,S+1|T^1_pq|S,S>`` in the MO basis."""
        s = ket.spin
        cidx, oidx, vidx, nmo = self.orbitals
        gamma = np.zeros((nmo, nmo))
        b_cv = bra.amplitude
        k_oo, k_co, k_cv, k_ov, k_cv0 = ket.amplitude
        f_cvcv = np.sqrt(s / (2.0 * (s + 1.0)))

        # CV(ai)-OO, -CO(vj), and -OV(bv).
        self._add(gamma, vidx, cidx, -k_oo * b_cv.T)
        self._add(gamma, vidx, oidx, lib.einsum('ia,iv->av', b_cv, k_co))
        self._add(gamma, oidx, cidx, lib.einsum('ia,va->vi', b_cv, k_ov))

        # CV(ai)-CV(bj).
        self._add(gamma, cidx, cidx, f_cvcv * lib.einsum('ia,ja->ji', b_cv, k_cv))
        self._add(gamma, vidx, vidx, f_cvcv * lib.einsum('ia,ib->ab', b_cv, k_cv))

        # CV(ai)-CV0(bj): opposite signs for the core and virtual blocks.
        factor = 1.0 / np.sqrt(2.0)
        self._add(gamma, cidx, cidx, -factor * lib.einsum('ia,ja->ji', b_cv, k_cv0))
        self._add(gamma, vidx, vidx, factor * lib.einsum('ia,ib->ab', b_cv, k_cv0))
        return gamma

    def _gamma_delta0_delta0(self, bra, ket):
        s = bra.spin
        cidx, oidx, vidx, nmo = self.orbitals
        gamma = np.zeros((nmo, nmo))
        b_oo, b_co, b_cv, b_ov, b_cv0 = bra.amplitude
        k_oo, k_co, k_cv, k_ov, k_cv0 = ket.amplitude
        rt2 = np.sqrt(2.0)
        f_oo_cv = np.sqrt(s / (s + 1.0)) if s > 0 else 0.0
        f_mix = -(s - 1.0) / (2.0 * np.sqrt(s * (s + 1.0))) if s > 0 else 0.0
        f_cvcv = 1.0 / (rt2 * (s + 1.0))
        f_cvcv0 = np.sqrt(s / (2.0 * (s + 1.0))) if s > 0 else 0.0

        self._add(gamma, cidx, oidx, b_oo * k_co / rt2)
        self._add(gamma, oidx, vidx, b_oo * k_ov / rt2)
        self._add(gamma, cidx, vidx, b_oo * k_cv * f_oo_cv)
        self._add(gamma, oidx, cidx, k_oo * b_co.T / rt2)
        self._add(gamma, cidx, cidx, lib.einsum('iu,ju->ji', b_co, k_co) / rt2)
        self._add(gamma, oidx, oidx, -lib.einsum('iu,iv->uv', b_co, k_co) / rt2)
        self._add(gamma, oidx, vidx, f_mix * lib.einsum('iu,ib->ub', b_co, k_cv))
        self._add(gamma, oidx, vidx, -0.5 * lib.einsum('iu,ib->ub', b_co, k_cv0))
        self._add(gamma, vidx, oidx, k_oo * b_ov.T / rt2)
        self._add(gamma, vidx, vidx, lib.einsum('ua,ub->ab', b_ov, k_ov) / rt2)
        self._add(gamma, oidx, oidx, -lib.einsum('ua,va->vu', b_ov, k_ov) / rt2)
        self._add(gamma, cidx, oidx, f_mix * lib.einsum('ua,ja->ju', b_ov, k_cv))
        self._add(gamma, cidx, oidx, 0.5 * lib.einsum('ua,ja->ju', b_ov, k_cv0))
        self._add(gamma, vidx, cidx, k_oo * b_cv.T * f_oo_cv)
        self._add(gamma, vidx, oidx, f_mix * lib.einsum('ia,iv->av', b_cv, k_co))
        self._add(gamma, oidx, cidx, f_mix * lib.einsum('ia,va->vi', b_cv, k_ov))
        self._add(gamma, cidx, cidx, f_cvcv * lib.einsum('ia,ja->ji', b_cv, k_cv))
        self._add(gamma, vidx, vidx, f_cvcv * lib.einsum('ia,ib->ab', b_cv, k_cv))
        self._add(gamma, cidx, cidx, f_cvcv0 * lib.einsum('ia,ja->ji', b_cv, k_cv0))
        self._add(gamma, vidx, vidx, -f_cvcv0 * lib.einsum('ia,ib->ab', b_cv, k_cv0))
        self._add(gamma, vidx, oidx, -0.5 * lib.einsum('ia,iv->av', b_cv0, k_co))
        self._add(gamma, oidx, cidx, 0.5 * lib.einsum('ia,va->vi', b_cv0, k_ov))
        self._add(gamma, cidx, cidx, f_cvcv0 * lib.einsum('ia,ja->ji', b_cv0, k_cv))
        self._add(gamma, vidx, vidx, -f_cvcv0 * lib.einsum('ia,ib->ab', b_cv0, k_cv))
        return gamma

    def _gamma_delta0_deltam1(self, bra, ket):
        s = bra.spin
        cidx, oidx, vidx, nmo = self.orbitals
        gamma = np.zeros((nmo, nmo))
        b_oo, b_co, b_cv, b_ov, b_cv0 = bra.amplitude
        k_co, k_cv, k_oo, k_ov = ket.amplitude
        f1 = np.sqrt((2.0 * s - 1.0) / (2.0 * s))
        f2 = np.sqrt((2.0 * s - 1.0) / (2.0 * s + 1.0))
        f_co_cv = f2 / (2.0 * s)
        f_cv_oo = 1.0 / np.sqrt(2.0 * s * (s + 1.0))
        f_cv_co = np.sqrt((s + 1.0) * (2.0 * s - 1.0)) / (2.0 * s)
        f_cv_cv = np.sqrt((s + 1.0) * (2.0 * s - 1.0) / (2.0 * s * (2.0 * s + 1.0)))
        f_cv0 = np.sqrt((2.0 * s - 1.0) / (4.0 * s))
        f_cv0_cv = np.sqrt((2.0 * s - 1.0) / (2.0 * (2.0 * s + 1.0)))

        self._add(gamma, oidx, oidx, b_oo * k_oo)
        self._add(gamma, cidx, oidx, b_oo * k_co * f1)
        self._add(gamma, oidx, vidx, b_oo * k_ov * f1)
        self._add(gamma, cidx, vidx, b_oo * k_cv * f2)
        self._add(gamma, oidx, cidx, lib.einsum('iu,wu->wi', b_co, k_oo))
        self._add(gamma, cidx, cidx, f1 * lib.einsum('iu,ju->ji', b_co, k_co))
        self._add(gamma, oidx, oidx, lib.einsum('iu,iv->uv', b_co, k_co) / np.sqrt(2.0 * s * (2.0 * s - 1.0)))
        self._add(gamma, oidx, vidx, f_co_cv * lib.einsum('iu,ib->ub', b_co, k_cv))
        self._add(gamma, vidx, oidx, lib.einsum('ua,uv->av', b_ov, k_oo))
        self._add(gamma, vidx, vidx, f1 * lib.einsum('ua,ub->ab', b_ov, k_ov))
        self._add(gamma, oidx, oidx, lib.einsum('ua,va->vu', b_ov, k_ov) / np.sqrt(2.0 * s * (2.0 * s - 1.0)))
        self._add(gamma, cidx, oidx, f_co_cv * lib.einsum('ua,ja->ju', b_ov, k_cv))
        tr_oo = np.trace(k_oo)
        self._add(gamma, vidx, cidx, f_cv_oo * tr_oo * b_cv.T)
        self._add(gamma, vidx, oidx, f_cv_co * lib.einsum('ia,iv->av', b_cv, k_co))
        self._add(gamma, oidx, cidx, f_cv_co * lib.einsum('ia,va->vi', b_cv, k_ov))
        self._add(gamma, cidx, cidx, f_cv_cv * lib.einsum('ia,ja->ji', b_cv, k_cv))
        self._add(gamma, vidx, vidx, f_cv_cv * lib.einsum('ia,ib->ab', b_cv, k_cv))
        self._add(gamma, vidx, oidx, -f_cv0 * lib.einsum('ia,iv->av', b_cv0, k_co))
        self._add(gamma, oidx, cidx, f_cv0 * lib.einsum('ia,va->vi', b_cv0, k_ov))
        self._add(gamma, cidx, cidx, f_cv0_cv * lib.einsum('ia,ja->ji', b_cv0, k_cv))
        self._add(gamma, vidx, vidx, -f_cv0_cv * lib.einsum('ia,ib->ab', b_cv0, k_cv))
        return gamma

    def _gamma_deltam1_deltam1(self, bra, ket):
        s = bra.spin + 1.0
        cidx, oidx, vidx, nmo = self.orbitals
        gamma = np.zeros((nmo, nmo))
        b_co, b_cv, b_oo, b_ov = bra.amplitude
        k_co, k_cv, k_oo, k_ov = ket.amplitude
        rt2 = np.sqrt(2.0)
        c_oo_mix = -(s - 1.0) / np.sqrt(s * (2.0 * s - 1.0))
        c_oo_tr = -1.0 / (2.0 * np.sqrt(s * (2.0 * s - 1.0)))
        c_oo_cv = -np.sqrt((2.0 * s - 1.0) / (2.0 * s + 1.0)) / (rt2 * s)
        c_coco_1 = -((s - 1.0) * (2.0 * s + 1.0)) / (rt2 * s * (2.0 * s - 1.0))
        c_coco_2 = -(s - 1.0) / (rt2 * s)
        c_cv_mix = -((s - 1.0) / (2.0 * s)) * np.sqrt((2.0 * s + 1.0) / s)
        c_cvcv = -(s - 1.0) / (rt2 * s)
        tr_boo = np.trace(b_oo)
        tr_koo = np.trace(k_oo)

        self._add(gamma, oidx, oidx, -lib.einsum('ut,wt->wu', b_oo, k_oo) / rt2)
        self._add(gamma, oidx, oidx, -lib.einsum('ut,uv->tv', b_oo, k_oo) / rt2)
        self._add(gamma, cidx, oidx, c_oo_mix * lib.einsum('ut,jt->ju', b_oo, k_co))
        self._add(gamma, cidx, oidx, c_oo_tr * tr_boo * k_co)
        self._add(gamma, oidx, vidx, c_oo_mix * lib.einsum('ut,ub->tb', b_oo, k_ov))
        self._add(gamma, oidx, vidx, c_oo_tr * tr_boo * k_ov)
        self._add(gamma, cidx, vidx, c_oo_cv * tr_boo * k_cv)
        self._add(gamma, oidx, cidx, c_oo_mix * lib.einsum('iu,wu->wi', b_co, k_oo))
        self._add(gamma, oidx, cidx, c_oo_tr * tr_koo * b_co.T)
        self._add(gamma, oidx, oidx, c_coco_1 * lib.einsum('iu,iv->uv', b_co, k_co))
        self._add(gamma, cidx, cidx, c_coco_2 * lib.einsum('iu,ju->ji', b_co, k_co))
        self._add(gamma, oidx, vidx, c_cv_mix * lib.einsum('iu,ib->ub', b_co, k_cv))
        self._add(gamma, vidx, oidx, c_oo_mix * lib.einsum('ua,uv->av', b_ov, k_oo))
        self._add(gamma, vidx, oidx, c_oo_tr * tr_koo * b_ov.T)
        self._add(gamma, oidx, oidx, c_coco_1 * lib.einsum('ua,va->vu', b_ov, k_ov))
        self._add(gamma, vidx, vidx, c_coco_2 * lib.einsum('ua,ub->ab', b_ov, k_ov))
        self._add(gamma, cidx, oidx, c_cv_mix * lib.einsum('ua,ja->ju', b_ov, k_cv))
        self._add(gamma, vidx, cidx, c_oo_cv * tr_koo * b_cv.T)
        self._add(gamma, vidx, oidx, c_cv_mix * lib.einsum('ia,iv->av', b_cv, k_co))
        self._add(gamma, oidx, cidx, c_cv_mix * lib.einsum('ia,va->vi', b_cv, k_ov))
        self._add(gamma, cidx, cidx, c_cvcv * lib.einsum('ia,ja->ji', b_cv, k_cv))
        self._add(gamma, vidx, vidx, c_cvcv * lib.einsum('ia,ib->ab', b_cv, k_cv))
        return gamma

    def reduced_transition_density(self, bra, ket):
        if abs(bra.spin - ket.spin) > 1 + 1e-12:
            return np.zeros((self._scf.mol.nao_nr(), self._scf.mol.nao_nr()))
        if abs(bra.spin - ket.spin) < 1e-12:
            if abs(bra.spin) < 1e-12:
                gamma_mo = np.zeros((self.orbitals[3], self.orbitals[3]))
            elif bra.delta_s == 1 and ket.delta_s == 1:
                coefficient = clebsch_gordan_rank1(bra.spin, bra.spin, 0, bra.spin, bra.spin)
                gamma_mo = self._gamma_deltap1_deltap1(bra, ket) / coefficient
            elif bra.delta_s == 0 and ket.delta_s == 0:
                coefficient = clebsch_gordan_rank1(bra.spin, bra.spin, 0, bra.spin, bra.spin)
                gamma_mo = self._gamma_delta0_delta0(bra, ket) / coefficient
            elif bra.delta_s == -1 and ket.delta_s == -1:
                coefficient = clebsch_gordan_rank1(bra.spin, bra.spin, 0, bra.spin, bra.spin)
                gamma_mo = self._gamma_deltam1_deltam1(bra, ket) / coefficient
            else:
                raise NotImplementedError('NTTDA SOC supports matching deltaS blocks only')
        elif abs(bra.spin - ket.spin - 1) < 1e-12:
            if bra.delta_s == 1 and ket.delta_s == 0:
                gamma_mo = self._gamma_deltap1_delta0(bra, ket)
            elif bra.delta_s == 0 and ket.delta_s == -1:
                gamma_mo = self._gamma_delta0_deltam1(bra, ket)
            else:
                raise NotImplementedError('Unsupported NTTDA SOC transition block')
            coefficient = clebsch_gordan_rank1(ket.spin, ket.spin, 1, bra.spin, bra.spin)
            gamma_mo /= coefficient
        else:
            raise NotImplementedError('Use the Hermitian-conjugate order for lowering spin blocks')

        coeff = np.asarray(self._scf.mo_coeff)
        # gamma_mo[p, q] = <bra||T_{pq}||ket>, which should be transposed
        return lib.einsum('up,pq,vq->uv', coeff.conj(), gamma_mo, coeff).T


__all__ = ['SOC']
