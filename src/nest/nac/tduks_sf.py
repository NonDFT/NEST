#!/usr/bin/env python
# Copyright 2014-2024 The PySCF Developers. All Rights Reserved.
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
#
# Ref:
# J. Chem. Theory Comput. 2026, 22, 6076
#

import copy
from functools import reduce

import numpy as np

from pyscf import gto, lib
from pyscf.lib import logger
from pyscf.scf import ucphf
from pyscf.grad import tdrhf as tdrhf_grad
from pyscf.grad import tduks as tduks_grad
from nest.grad import tduks_sf as tduks_sf_grad


def get_Hellmann_Feymann(td_grad, x_y_I, x_y_J, atmlst=None, max_memory=6000, verbose=logger.INFO,
                         state_I=None, state_J=None):
    """Return the energy-weighted, ETF-corrected coupling between two SF states."""
    return _nac_terms(td_grad, x_y_I, x_y_J, atmlst, max_memory, verbose)[0]


def nac_csf(td_grad, x_y_I, x_y_J, atmlst=None):
    """Return the CSF overlap-derivative contribution without solving for the Z-vector."""
    return _nac_terms(td_grad, x_y_I, x_y_J, atmlst, with_response=False)[1]


def _nac_terms(td_grad, x_y_I, x_y_J, atmlst, max_memory=6000, verbose=logger.INFO, with_response=True):
    """Return the Hellmann–Feynman numerator and the unscaled CSF contribution."""
    log = logger.new_logger(td_grad, verbose)
    time0 = logger.process_clock(), logger.perf_counter()

    mol = td_grad.mol
    mf = td_grad.base._scf
    mo_coeff = mf.mo_coeff
    mo_energy = mf.mo_energy
    mo_occ = mf.mo_occ
    occidxa = np.where(mo_occ[0] > 0)[0]
    occidxb = np.where(mo_occ[1] > 0)[0]
    viridxa = np.where(mo_occ[0] == 0)[0]
    viridxb = np.where(mo_occ[1] == 0)[0]
    nocca = len(occidxa)
    noccb = len(occidxb)
    nvira = len(viridxa)
    nvirb = len(viridxb)
    orboa = mo_coeff[0][:, occidxa]
    orbob = mo_coeff[1][:, occidxb]
    orbva = mo_coeff[0][:, viridxa]
    orbvb = mo_coeff[1][:, viridxb]
    nao = mo_coeff[0].shape[0]
    nmoa = nocca + nvira
    nmob = noccb + nvirb

    if td_grad.base.extype == 0:
        y_I, x_I = x_y_I
        y_J, x_J = x_y_J
        if not isinstance(x_I, np.ndarray):
            x_I = np.zeros((nocca, nvirb))
        if not isinstance(x_J, np.ndarray):
            x_J = np.zeros((nocca, nvirb))
    elif td_grad.base.extype == 1:
        x_I, y_I = x_y_I
        x_J, y_J = x_y_J
        if not isinstance(y_I, np.ndarray):
            y_I = np.zeros((noccb, nvira))
        if not isinstance(y_J, np.ndarray):
            y_J = np.zeros((noccb, nvira))

    dvva_IJ = lib.einsum('ia,ib->ab', y_I, y_J)
    dvvb_IJ = lib.einsum('ia,ib->ab', x_I, x_J)
    dooa_IJ = -lib.einsum('ia,ja->ij', x_I, x_J)
    doob_IJ = -lib.einsum('ia,ja->ij', y_I, y_J)
    dmzooa_IJ = reduce(np.dot, (orboa, dooa_IJ, orboa.T))
    dmzooa_IJ += reduce(np.dot, (orbva, dvva_IJ, orbva.T))
    dmzoob_IJ = reduce(np.dot, (orbob, doob_IJ, orbob.T))
    dmzoob_IJ += reduce(np.dot, (orbvb, dvvb_IJ, orbvb.T))
    # The antisymmetric density contributes only to the uncorrected NAC.
    dm_csf = dmzooa_IJ - dmzoob_IJ if td_grad.base.extype == 0 else dmzoob_IJ - dmzooa_IJ
    dm_csf = dm_csf.T - dm_csf
    mf_grad = mf.nuc_grad_method()
    s1 = mf_grad.get_ovlp(mol)
    if atmlst is None:
        atmlst = range(mol.natm)
    offsetdic = mol.offset_nr_by_atom()
    csf = np.empty((len(atmlst), 3))
    for k, ia in enumerate(atmlst):
        _, _, p0, p1 = offsetdic[ia]
        csf[k] = lib.einsum('xpq,pq->x', s1[:, p0:p1], dm_csf[p0:p1]) * 0.5
    if not with_response:
        return None, csf

    dmx_I = reduce(np.dot, (orbvb, x_I.T, orboa.T))
    dmy_I = reduce(np.dot, (orbob, y_I, orbva.T))
    dmx_J = reduce(np.dot, (orbvb, x_J.T, orboa.T))
    dmy_J = reduce(np.dot, (orbob, y_J, orbva.T))
    dmt_I = dmx_I + dmy_I
    dmt_J = dmx_J + dmy_J

    dmzooa = (dmzooa_IJ + dmzooa_IJ.T) * 0.5
    dmzoob = (dmzoob_IJ + dmzoob_IJ.T) * 0.5

    ni = mf._numint
    ni.libxc.test_deriv_order(mf.xc, 3, raise_error=True)
    omega, alpha, hyb = ni.rsh_and_hybrid_coeff(mf.xc, mol.spin)

    f1vo_I, f1oo, vxc1, k1ao = tduks_sf_grad._contract_xc_kernel(
        td_grad, mf.xc, dmt_I, (dmzooa, dmzoob), dmvo_2=dmt_J, max_memory=max_memory,
    )
    f1vo_J = tduks_sf_grad._contract_xc_kernel(
        td_grad, mf.xc, dmt_J, with_vxc=False, with_kxc=False, max_memory=max_memory,
    )[0]

    with_k = ni.libxc.is_hybrid_xc(mf.xc)
    if with_k:
        vj0, vk0 = mf.get_jk(mol, (dmzooa, dmzoob), hermi=1)
        vk1_I, vk1_J = mf.get_k(mol, (dmt_I, dmt_J), hermi=0) * hyb
        vk0 = vk0 * hyb
        if omega != 0:
            vk0 += mf.get_k(mol, (dmzooa, dmzoob), hermi=1, omega=omega) * (alpha - hyb)
            vk1_I_omega, vk1_J_omega = mf.get_k(mol, (dmt_I, dmt_J), hermi=0, omega=omega)
            vk1_I += vk1_I_omega * (alpha - hyb)
            vk1_J += vk1_J_omega * (alpha - hyb)

        veff0doo = vj0[0] + vj0[1] - vk0 + f1oo[:, 0] + k1ao[:, 0]
        wvoa = reduce(np.dot, (orbva.T, veff0doo[0], orboa))
        wvob = reduce(np.dot, (orbvb.T, veff0doo[1], orbob))

        veff0mo_I = reduce(np.dot, (mo_coeff[1].T, f1vo_I[0] - vk1_I, mo_coeff[0]))
        veff0mo_J = reduce(np.dot, (mo_coeff[1].T, f1vo_J[0] - vk1_J, mo_coeff[0]))
        wvoa += lib.einsum('ac,ka->ck', veff0mo_I[noccb:, nocca:], x_J) * 0.5
        wvoa += lib.einsum('ac,ka->ck', veff0mo_J[noccb:, nocca:], x_I) * 0.5
        wvoa -= lib.einsum('jk,jc->ck', veff0mo_I[:noccb, :nocca], y_J) * 0.5
        wvoa -= lib.einsum('jk,jc->ck', veff0mo_J[:noccb, :nocca], y_I) * 0.5
        wvob += lib.einsum('ac,ka->ck', veff0mo_I.T[nocca:, noccb:], y_J) * 0.5
        wvob += lib.einsum('ac,ka->ck', veff0mo_J.T[nocca:, noccb:], y_I) * 0.5
        wvob -= lib.einsum('jk,jc->ck', veff0mo_I.T[:nocca, :noccb], x_J) * 0.5
        wvob -= lib.einsum('jk,jc->ck', veff0mo_J.T[:nocca, :noccb], x_I) * 0.5

    else:
        vj0 = mf.get_j(mol, (dmzooa, dmzoob), hermi=1)
        veff0doo = vj0[0] + vj0[1] + f1oo[:, 0] + k1ao[:, 0]
        wvoa = reduce(np.dot, (orbva.T, veff0doo[0], orboa))
        wvob = reduce(np.dot, (orbvb.T, veff0doo[1], orbob))

        veff0mo_I = reduce(np.dot, (mo_coeff[1].T, f1vo_I[0], mo_coeff[0]))
        veff0mo_J = reduce(np.dot, (mo_coeff[1].T, f1vo_J[0], mo_coeff[0]))
        wvoa += lib.einsum('ac,ka->ck', veff0mo_I[noccb:, nocca:], x_J) * 0.5
        wvoa += lib.einsum('ac,ka->ck', veff0mo_J[noccb:, nocca:], x_I) * 0.5
        wvoa -= lib.einsum('jk,jc->ck', veff0mo_I[:noccb, :nocca], y_J) * 0.5
        wvoa -= lib.einsum('jk,jc->ck', veff0mo_J[:noccb, :nocca], y_I) * 0.5
        wvob += lib.einsum('ac,ka->ck', veff0mo_I.T[nocca:, noccb:], y_J) * 0.5
        wvob += lib.einsum('ac,ka->ck', veff0mo_J.T[nocca:, noccb:], y_I) * 0.5
        wvob -= lib.einsum('jk,jc->ck', veff0mo_I.T[:nocca, :noccb], x_J) * 0.5
        wvob -= lib.einsum('jk,jc->ck', veff0mo_J.T[:nocca, :noccb], x_I) * 0.5

    vresp = mf.gen_response(hermi=1)

    def fvind(x):
        xa = x[0, : nvira * nocca].reshape(nvira, nocca)
        xb = x[0, nvira * nocca :].reshape(nvirb, noccb)
        dma = reduce(np.dot, (orbva, xa, orboa.T))
        dmb = reduce(np.dot, (orbvb, xb, orbob.T))
        dm1 = np.stack((dma + dma.T, dmb + dmb.T))
        v1 = vresp(dm1)
        v1a = reduce(np.dot, (orbva.T, v1[0], orboa))
        v1b = reduce(np.dot, (orbvb.T, v1[1], orbob))
        return np.hstack((v1a.ravel(), v1b.ravel()))

    z1a, z1b = ucphf.solve(
        fvind, mo_energy, mo_occ, (wvoa, wvob), max_cycle=td_grad.cphf_max_cycle, tol=td_grad.cphf_conv_tol
    )[0]
    time1 = log.timer('Z-vector using UCPHF solver', *time0)

    z1ao = np.empty((2, nao, nao))
    z1ao[0] = reduce(np.dot, (orbva, z1a, orboa.T))
    z1ao[1] = reduce(np.dot, (orbvb, z1b, orbob.T))
    veff = vresp((z1ao + z1ao.transpose(0, 2, 1)))

    im0a = np.zeros((nmoa, nmoa))
    im0b = np.zeros((nmob, nmob))
    im0a[:nocca, :nocca] = reduce(np.dot, (orboa.T, veff0doo[0] + veff[0], orboa))
    im0b[:noccb, :noccb] = reduce(np.dot, (orbob.T, veff0doo[1] + veff[1], orbob))
    im0a[:nocca, :nocca] += lib.einsum('al,ka->lk', veff0mo_I[noccb:, :nocca], x_J) * 0.5
    im0a[:nocca, :nocca] += lib.einsum('al,ka->lk', veff0mo_J[noccb:, :nocca], x_I) * 0.5

    im0b[:noccb, :noccb] += lib.einsum('al,ka->lk', veff0mo_I.T[nocca:, :noccb], y_J) * 0.5
    im0b[:noccb, :noccb] += lib.einsum('al,ka->lk', veff0mo_J.T[nocca:, :noccb], y_I) * 0.5

    im0a[nocca:, nocca:] = lib.einsum('jd,jc->dc', veff0mo_I[:noccb, nocca:], y_J) * 0.5
    im0a[nocca:, nocca:] += lib.einsum('jd,jc->dc', veff0mo_J[:noccb, nocca:], y_I) * 0.5

    im0b[noccb:, noccb:] = lib.einsum('jd,jc->dc', veff0mo_I.T[:nocca, noccb:], x_J) * 0.5
    im0b[noccb:, noccb:] += lib.einsum('jd,jc->dc', veff0mo_J.T[:nocca, noccb:], x_I) * 0.5

    im0a[:nocca, nocca:] = lib.einsum('jk,jc->kc', veff0mo_I[:noccb, :nocca], y_J)
    im0a[:nocca, nocca:] += lib.einsum('jk,jc->kc', veff0mo_J[:noccb, :nocca], y_I)
    im0b[:noccb, noccb:] = lib.einsum('jk,jc->kc', veff0mo_I.T[:nocca, :noccb], x_J)
    im0b[:noccb, noccb:] += lib.einsum('jk,jc->kc', veff0mo_J.T[:nocca, :noccb], x_I)

    zeta_a = (mo_energy[0][:, None] + mo_energy[0]) * 0.5
    zeta_b = (mo_energy[1][:, None] + mo_energy[1]) * 0.5
    zeta_a[nocca:, :nocca] = mo_energy[0][:nocca]
    zeta_b[noccb:, :noccb] = mo_energy[1][:noccb]
    zeta_a[:nocca, nocca:] = mo_energy[0][nocca:]
    zeta_b[:noccb, noccb:] = mo_energy[1][noccb:]
    dm1a = np.zeros((nmoa, nmoa))
    dm1b = np.zeros((nmob, nmob))
    dm1a[:nocca, :nocca] = (dooa_IJ + dooa_IJ.T) * 0.5
    dm1b[:noccb, :noccb] = (doob_IJ + doob_IJ.T) * 0.5
    dm1a[nocca:, nocca:] = (dvva_IJ + dvva_IJ.T) * 0.5
    dm1b[noccb:, noccb:] = (dvvb_IJ + dvvb_IJ.T) * 0.5
    dm1a[nocca:, :nocca] = z1a * 2
    dm1b[noccb:, :noccb] = z1b * 2

    im0a = reduce(np.dot, (mo_coeff[0], im0a + zeta_a * dm1a, mo_coeff[0].T))
    im0b = reduce(np.dot, (mo_coeff[1], im0b + zeta_b * dm1b, mo_coeff[1].T))
    im0 = im0a + im0b

    hcore_deriv = mf_grad.hcore_generator(mol)

    dmz1dooa = 4 * z1ao[0] + 2 * dmzooa
    dmz1doob = 4 * z1ao[1] + 2 * dmzoob
    oo0a = reduce(np.dot, (orboa, orboa.T))
    oo0b = reduce(np.dot, (orbob, orbob.T))
    as_dm1 = (dmz1dooa + dmz1doob) * 0.5

    if with_k:
        dm = (oo0a, dmz1dooa + dmz1dooa.T, oo0b, dmz1doob + dmz1doob.T)
        vj, vk = td_grad.get_jk(mol, dm, hermi=1)
        vj = vj.reshape(2, 2, 3, nao, nao)
        vk = vk.reshape(2, 2, 3, nao, nao) * hyb
        vk1_all = -td_grad.get_k(mol, (dmt_I, dmt_I.T, dmt_J, dmt_J.T)) * hyb
        vk1_I = vk1_all[:2]
        vk1_J = vk1_all[2:]
        if omega != 0:
            vk += td_grad.get_k(mol, dm, omega=omega).reshape(2, 2, 3, nao, nao) * (
                alpha - hyb
            )
            vk1_all_omega = -td_grad.get_k(mol, (dmt_I, dmt_I.T, dmt_J, dmt_J.T), omega=omega)
            vk1_I += vk1_all_omega[:2] * (alpha - hyb)
            vk1_J += vk1_all_omega[2:] * (alpha - hyb)
        veff1 = vj[0] + vj[1] - vk
    else:
        dm = (oo0a, dmz1dooa + dmz1dooa.T, oo0b, dmz1doob + dmz1doob.T)
        vj = td_grad.get_j(mol, dm, hermi=1).reshape(2, 2, 3, nao, nao)
        veff1 = vj[0] + vj[1]
        veff1 = np.stack((veff1, veff1))

    fxcz1 = tduks_grad._contract_xc_kernel(td_grad, mf.xc, 2 * z1ao, None, False, False, max_memory)[0]
    veff1[:, 0] += vxc1[:, 1:]
    veff1[:, 1] += (f1oo[:, 1:] + fxcz1[:, 1:] + k1ao[:, 1:]) * 4
    veff1a, veff1b = veff1
    time1 = log.timer('2e AO integral derivatives', *time1)

    de = np.zeros((len(atmlst), 3))

    for k, ia in enumerate(atmlst):
        shl0, shl1, p0, p1 = offsetdic[ia]

        h1ao = hcore_deriv(ia)
        de[k] = lib.einsum('xpq,pq->x', h1ao, as_dm1)

        de[k] -= lib.einsum('xpq,pq->x', s1[:, p0:p1], im0[p0:p1])
        de[k] -= lib.einsum('xqp,pq->x', s1[:, p0:p1], im0[:, p0:p1])

        de[k] += lib.einsum('xpq,pq->x', veff1a[0, :, p0:p1], dmz1dooa[p0:p1]) * 0.5
        de[k] += lib.einsum('xpq,pq->x', veff1b[0, :, p0:p1], dmz1doob[p0:p1]) * 0.5
        de[k] += lib.einsum('xpq,qp->x', veff1a[0, :, p0:p1], dmz1dooa[:, p0:p1]) * 0.5
        de[k] += lib.einsum('xpq,qp->x', veff1b[0, :, p0:p1], dmz1doob[:, p0:p1]) * 0.5
        de[k] += lib.einsum('xij,ij->x', veff1a[1, :, p0:p1], oo0a[p0:p1]) * 0.5
        de[k] += lib.einsum('xij,ij->x', veff1b[1, :, p0:p1], oo0b[p0:p1]) * 0.5

        if td_grad.base.collinear == 'mcol':
            de[k] += lib.einsum('xpq,pq->x', f1vo_I[1:, p0:p1], dmt_J[p0:p1])
            de[k] += lib.einsum('xpq,pq->x', f1vo_J[1:, p0:p1], dmt_I[p0:p1])
            de[k] += lib.einsum('xpq,pq->x', f1vo_I[1:, p0:p1], dmt_J.T[p0:p1])
            de[k] += lib.einsum('xpq,pq->x', f1vo_J[1:, p0:p1], dmt_I.T[p0:p1])

        if with_k:
            de[k] += lib.einsum('xpq,pq->x', vk1_I[0, :, p0:p1], dmt_J[p0:p1])
            de[k] += lib.einsum('xpq,pq->x', vk1_J[0, :, p0:p1], dmt_I[p0:p1])
            de[k] += lib.einsum('xpq,pq->x', vk1_I[1, :, p0:p1], dmt_J.T[p0:p1])
            de[k] += lib.einsum('xpq,pq->x', vk1_J[1, :, p0:p1], dmt_I.T[p0:p1])

    log.timer('SF-TDDFT nonadiabatic coupling', *time0)
    return de, csf


def as_scanner(nac):
    """Return a scanner that evaluates this NAC at each supplied geometry."""
    if isinstance(nac, lib.GradScanner):
        return nac
    name = nac.__class__.__name__ + NACScanner.__name_mixin__
    return lib.set_class(NACScanner(nac), (NACScanner, nac.__class__), name)


class NACScanner(lib.GradScanner):
    def __call__(self, mol_or_geom, **kwargs):
        if isinstance(mol_or_geom, gto.MoleBase):
            mol = mol_or_geom
        else:
            mol = self.mol.set_geom_(mol_or_geom, inplace=False)
        if mol.nao != self.mol.nao or mol.nelec != self.mol.nelec:
            self.reset_phase()
        self.base(mol)
        self.mol = mol
        return self.kernel(**kwargs)

    @property
    def converged(self):
        if self.state_I is None or self.state_J is None or self.base.converged is None:
            return False
        return (self.base._scf.converged and self.base.converged[self.state_I - 1]
                and self.base.converged[self.state_J - 1])


def _dot_amplitude_block(a, b):
    if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
        return np.dot(a.ravel(), b.ravel())
    return 0.0


def _scale_amplitude_block(a, factor):
    if isinstance(a, np.ndarray):
        return a * factor
    return a


class NonAdiabaticCouplings(tdrhf_grad.Gradients):
    cphf_max_cycle = tdrhf_grad.Gradients.cphf_max_cycle + 20

    def __init__(self, td):
        super().__init__(td)
        self.state_I = None
        self.state_J = None
        self.ediff = False
        self.use_etfs = False
        self.x_y_I_prev = None
        self.x_y_J_prev = None
        self._phase_states = None
        self._keys = self._keys.union({'state_I', 'state_J', 'ediff', 'use_etfs'})

    def dump_flags(self, verbose=None):
        super().dump_flags(verbose)
        log = logger.new_logger(self, verbose)
        log.info('State I = %s', self.state_I)
        log.info('State J = %s', self.state_J)
        log.info('ediff = %s', self.ediff)
        log.info('use_etfs = %s', self.use_etfs)
        return self

    def _vector_dot(self, vec1, vec2):
        x1, y1 = vec1
        x2, y2 = vec2
        dot_x = _dot_amplitude_block(x1, x2)
        dot_y = _dot_amplitude_block(y1, y2)
        return dot_x - dot_y

    def _scale_vector(self, vec, factor):
        x, y = vec
        return (_scale_amplitude_block(x, factor), _scale_amplitude_block(y, factor))

    def compute_nac(self, state_I=None, state_J=None, atmlst=None, ediff=None, use_etfs=None, use_cache=True):
        state_I = state_I if state_I is not None else self.state_I
        state_J = state_J if state_J is not None else self.state_J
        ediff = self.ediff if ediff is None else ediff
        use_etfs = self.use_etfs if use_etfs is None else use_etfs

        if state_I is None or state_J is None:
            raise RuntimeError('state_I and state_J must be specified')
        if self.base.xy is None:
            raise RuntimeError('Run the spin-flip TD calculation before requesting a NAC')
        nstates = len(self.base.xy)
        for state in (state_I, state_J):
            if not isinstance(state, (int, np.integer)) or isinstance(state, bool) or not 1 <= state <= nstates:
                raise ValueError(f'Expected an excited-state index from 1 to {nstates}, got {state!r}')
        if state_I == state_J:
            raise ValueError('state_I and state_J must be different')

        raw_x_y_I = self.base.xy[state_I - 1]
        raw_x_y_J = self.base.xy[state_J - 1]
        e_I = self.base.e[state_I - 1]
        e_J = self.base.e[state_J - 1]

        if self._phase_states == (state_I, state_J):
            if self._vector_dot(raw_x_y_I, self.x_y_I_prev) < 0:
                raw_x_y_I = self._scale_vector(raw_x_y_I, -1.0)
            if self._vector_dot(raw_x_y_J, self.x_y_J_prev) < 0:
                raw_x_y_J = self._scale_vector(raw_x_y_J, -1.0)

        hf_term, csf = _nac_terms(self, raw_x_y_I, raw_x_y_J, atmlst)
        self._phase_states = (state_I, state_J)
        self.x_y_I_prev = copy.deepcopy(raw_x_y_I)
        self.x_y_J_prev = copy.deepcopy(raw_x_y_J)
        delta_e = e_J - e_I
        nac = hf_term if use_etfs else hf_term + csf * delta_e
        if ediff:
            if abs(delta_e) < 1e-10:
                logger.warn(self, 'Energy difference is very small: %s. NAC not divided.', delta_e)
            else:
                nac = nac / delta_e
        return nac

    def kernel(self, state_I=None, state_J=None, atmlst=None, ediff=None, use_etfs=None, use_cache=True):
        if atmlst is None:
            atmlst = self.atmlst
        self.nac = self.compute_nac(state_I, state_J, atmlst, ediff, use_etfs, use_cache)
        if state_I is not None:
            self.state_I = state_I
        if state_J is not None:
            self.state_J = state_J
        self.atmlst = atmlst
        if ediff is not None:
            self.ediff = ediff
        if use_etfs is not None:
            self.use_etfs = use_etfs

        return self.nac

    def reset_phase(self):
        self._phase_states = None
        self.x_y_I_prev = None
        self.x_y_J_prev = None
        return self

    as_scanner = as_scanner


NAC = NonAdiabaticCouplings
