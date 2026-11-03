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

"""Direct spin-orbit-coupled TDA for a closed-shell RHF/RKS reference.

The trial-vector block order is ``[S, T-1, T0, T+1]``.  If the closed-shell
reference is included, its scalar coefficient precedes the four excitation
blocks.  SOC is added directly to the Casida A matrix instead of coupling a
truncated set of previously converged spin-free states.
"""

import numpy as np
from pyscf import lib, scf
from pyscf.lib import logger
from pyscf.tdscf import rhf

from nest._lr_eig import eigh as lr_eigh
from nest.soc.soc_ao import get_ao_soc


def _get_triplet_a(mf, frozen_mask):
    """Build the closed-shell triplet TDA matrix with the SF-TDA kernel."""
    from nest.sftda.uhf_sf import get_ab_sf

    mf_sf = mf.to_uks() if isinstance(mf, scf.hf.KohnShamDFT) else mf.to_uhf()
    mo_energy = np.asarray(mf_sf.mo_energy)[:, frozen_mask]
    mo_coeff = np.asarray(mf_sf.mo_coeff)[:, :, frozen_mask]
    mo_occ = np.asarray(mf_sf.mo_occ)[:, frozen_mask]

    a_sf = get_ab_sf(
        mf_sf,
        mo_energy=mo_energy,
        mo_coeff=mo_coeff,
        mo_occ=mo_occ,
        collinear_samples=1,
    )[0][1]

    # get_ab_sf uses explicit UHF/UKS Fock oo/vv blocks when alpha and beta
    # orbitals are identical.  Replace that one-electron part with the
    # canonical restricted orbital gaps used by RHF/RKS TDA.gen_vind.
    occidx_a = mo_occ[0] == 1
    viridx_b = mo_occ[1] == 0
    orbo_a = mo_coeff[0][:, occidx_a]
    orbv_b = mo_coeff[1][:, viridx_b]
    fock_a, fock_b = mf_sf.get_fock()
    fock_oo = orbo_a.conj().T @ fock_a @ orbo_a
    fock_vv = orbv_b.conj().T @ fock_b @ orbv_b
    nocc = fock_oo.shape[0]
    nvir = fock_vv.shape[0]
    fock_part = lib.einsum('ij,ab->iajb', np.eye(nocc), fock_vv)
    fock_part -= lib.einsum('ab,ji->iajb', np.eye(nvir), fock_oo)

    e_ia = mo_energy[1, viridx_b][None, :] - mo_energy[0, occidx_a][:, None]
    orbital_gap = np.diag(e_ia.ravel()).reshape(nocc, nvir, nocc, nvir)
    return a_sf - fock_part + orbital_gap


class SOTDDFT(rhf.TDBase):
    """Spin-orbit-coupled, TDA-only TDHF/TDDFT.

    ``xy[root]`` is stored as ``(x, 0)``, where ``x`` is a one-dimensional
    complex vector of length ``4*nocc*nvir`` or ``4*nocc*nvir + 1``.
    """

    singlet = None
    soctype = 'SOMF'
    include_reference = True

    _keys = {'soctype', 'include_reference', 'soc_ao', 'soc_mo'}

    def __init__(self, mf, soctype='SOMF', include_reference=True, frozen=None):
        super().__init__(mf, frozen=frozen)
        self.soctype = soctype
        self.include_reference = include_reference
        self.soc_ao = None
        self.soc_mo = None

    def dump_flags(self, verbose=None):
        super().dump_flags(verbose)
        log = logger.new_logger(self, verbose)
        log.info('soctype = %s', self.soctype)
        log.info('include_reference = %s', self.include_reference)
        return self

    def check_sanity(self):
        super().check_sanity()
        mo_occ = np.asarray(self._scf.mo_occ)
        mo_coeff = np.asarray(self._scf.mo_coeff)
        if mo_occ.ndim != 1 or np.any((mo_occ != 0) & (mo_occ != 2)):
            raise ValueError('SOTDDFT requires a closed-shell RHF/RKS reference')
        if mo_coeff.ndim != 2 or not np.isrealobj(mo_coeff):
            raise ValueError('SOTDDFT requires real, spin-restricted MO coefficients')
        if self.wfnsym is not None:
            raise NotImplementedError('SOTDDFT does not support wfnsym because SOC mixes spatial irreps')
        return self

    @staticmethod
    def _apply_one_body(z_vv, z_oo, xs, hole_sign):
        r"""Apply ``delta_ij z_ab + hole_sign delta_ab z_ji`` to ``xs``."""
        particle = lib.einsum('ab,xib->xia', z_vv, xs)
        hole = lib.einsum('ji,xja->xia', z_oo, xs)
        return particle + hole_sign * hole

    def gen_vind(self, mf=None):
        """Generate the matrix-vector product for the SOC Casida A matrix."""
        if mf is None:
            mf = self._scf
        if mf is not self._scf:
            raise ValueError('gen_vind must use the SCF object associated with SOTDDFT')
        self.check_sanity()

        # Reuse PySCF's complete RHF/RKS TDA response, including the canonical
        # Fock energy differences and the HF/DFT response kernel.
        vind_s, hdiag_s = rhf._gen_tda_operation(self, singlet=True, wfnsym=None)
        vind_t, hdiag_t = rhf._gen_tda_operation(self, singlet=False, wfnsym=None)

        mask = self.get_frozen_mask()
        mo_coeff = np.asarray(mf.mo_coeff)[:, mask]
        mo_occ = np.asarray(mf.mo_occ)[mask]
        occidx = np.where(mo_occ == 2)[0]
        viridx = np.where(mo_occ == 0)[0]
        nocc = occidx.size
        nvir = viridx.size
        nov = nocc * nvir

        self.soc_ao = get_ao_soc(mf, self.soctype)
        self.soc_mo = lib.einsum(
            'up,xuv,vq->xpq', mo_coeff.conj(), self.soc_ao, mo_coeff, optimize=True,
        )

        # soc_ao uses spherical components [-1, 0, +1]:
        # z^- = z^x-i*z^y = 2 Z_-1, z^z = sqrt(2) Z_0,
        # z^+ = z^x+i*z^y = -2 Z_+1.
        z_minus = 2 * self.soc_mo[0]
        z_z = np.sqrt(2) * self.soc_mo[1]
        z_plus = -2 * self.soc_mo[2]

        z_minus_oo = z_minus[np.ix_(occidx, occidx)]
        z_minus_vv = z_minus[np.ix_(viridx, viridx)]
        z_plus_oo = z_plus[np.ix_(occidx, occidx)]
        z_plus_vv = z_plus[np.ix_(viridx, viridx)]
        z_z_oo = z_z[np.ix_(occidx, occidx)]
        z_z_vv = z_z[np.ix_(viridx, viridx)]

        offset = int(self.include_reference)
        dimension = offset + 4 * nov

        if self.include_reference:
            z_minus_vo = z_minus[np.ix_(viridx, occidx)].T.reshape(-1)
            z_z_vo = z_z[np.ix_(viridx, occidx)].T.reshape(-1)
            z_plus_vo = z_plus[np.ix_(viridx, occidx)].T.reshape(-1)
            reference_coupling = np.concatenate((
                np.zeros(nov, dtype=np.complex128),
                0.5 * z_plus_vo,
                z_z_vo / np.sqrt(2),
                -0.5 * z_minus_vo,
            ))

        def apply_spin_free(vind_spin, xs):
            # PySCF's RKS fxc routines reject complex density matrices.  The
            # spin-free kernel is real-linear for the real orbitals required
            # above, so apply it independently to both components.  Davidson
            # guesses can also contain entire zero spin blocks; do not build
            # AO response densities for them.
            def apply_nonzero(real_xs):
                result = np.zeros_like(real_xs)
                active = np.any(real_xs != 0, axis=1)
                if np.any(active):
                    result[active] = vind_spin(real_xs[active])
                return result

            if np.iscomplexobj(xs):
                return apply_nonzero(xs.real) + 1j * apply_nonzero(xs.imag)
            return apply_nonzero(xs)

        def vind(zs):
            zs = np.asarray(zs)
            if zs.ndim == 1:
                zs = zs[None, :]
            if zs.shape[1] != dimension:
                raise ValueError(f'Trial vectors have length {zs.shape[1]}, expected {dimension}')

            nvec = len(zs)
            amplitudes = zs[:, offset:].reshape(nvec, 4, nocc, nvir)
            result = np.zeros((nvec, dimension), dtype=np.result_type(zs, self.soc_mo))
            out = result[:, offset:].reshape(nvec, 4, nocc, nvir)

            singlet_input = amplitudes[:, 0].reshape(nvec, nov)
            out[:, 0] = apply_spin_free(vind_s, singlet_input).reshape(nvec, nocc, nvir)
            triplets = amplitudes[:, 1:].reshape(3 * nvec, nov)
            out[:, 1:] = apply_spin_free(vind_t, triplets).reshape(nvec, 3, nocc, nvir)

            singlet, triplet_m1, triplet_0, triplet_p1 = amplitudes.transpose(1, 0, 2, 3)
            d_minus_s = self._apply_one_body(z_minus_vv, z_minus_oo, singlet, -1)
            d_z_s = self._apply_one_body(z_z_vv, z_z_oo, singlet, -1)
            d_plus_s = self._apply_one_body(z_plus_vv, z_plus_oo, singlet, -1)
            q_minus_m1 = self._apply_one_body(z_minus_vv, z_minus_oo, triplet_m1, 1)
            q_minus_0 = self._apply_one_body(z_minus_vv, z_minus_oo, triplet_0, 1)
            q_plus_0 = self._apply_one_body(z_plus_vv, z_plus_oo, triplet_0, 1)
            q_plus_p1 = self._apply_one_body(z_plus_vv, z_plus_oo, triplet_p1, 1)
            q_z_m1 = self._apply_one_body(z_z_vv, z_z_oo, triplet_m1, 1)
            q_z_p1 = self._apply_one_body(z_z_vv, z_z_oo, triplet_p1, 1)

            out[:, 0] += (
                self._apply_one_body(z_minus_vv, z_minus_oo, triplet_m1, -1) / np.sqrt(8)
                + self._apply_one_body(z_z_vv, z_z_oo, triplet_0, -1) / 2
                - self._apply_one_body(z_plus_vv, z_plus_oo, triplet_p1, -1) / np.sqrt(8)
            )
            out[:, 1] += d_plus_s / np.sqrt(8) - q_z_m1 / 2 + q_plus_0 / np.sqrt(8)
            out[:, 2] += d_z_s / 2 + q_minus_m1 / np.sqrt(8) + q_plus_p1 / np.sqrt(8)
            out[:, 3] += -d_minus_s / np.sqrt(8) + q_minus_0 / np.sqrt(8) + q_z_p1 / 2

            if self.include_reference:
                result[:, 0] = lib.einsum('p,xp->x', reference_coupling.conj(), zs[:, 1:])
                result[:, 1:] += zs[:, :1] * reference_coupling
            return result

        hdiag = np.concatenate((hdiag_s, hdiag_t, hdiag_t, hdiag_t)).astype(np.complex128)
        q_z_diag = (
            z_z_vv.diagonal()[None, :] + z_z_oo.diagonal()[:, None]
        ).reshape(-1)
        hdiag[nov:2 * nov] -= q_z_diag / 2
        hdiag[3 * nov:4 * nov] += q_z_diag / 2
        if self.include_reference:
            hdiag = np.concatenate((np.zeros(1, dtype=np.complex128), hdiag))
        return vind, hdiag

    def get_ab(self, mf=None):
        """Return the complete SOC-TDA Hamiltonian in the excitation basis.

        Unlike the regular PySCF ``get_ab``, this TDA-only method returns one
        two-dimensional matrix rather than an ``(A, B)`` tuple.  Its basis is
        ``[reference?, S, T-1, T0, T+1]``.
        """
        if mf is None:
            mf = self._scf
        if mf is not self._scf:
            raise ValueError('get_ab must use the SCF object associated with SOTDDFT')
        self.check_sanity()

        mask = self.get_frozen_mask()
        mo_coeff = np.asarray(mf.mo_coeff)[:, mask]
        mo_occ = np.asarray(mf.mo_occ)[mask]
        occidx = np.where(mo_occ == 2)[0]
        viridx = np.where(mo_occ == 0)[0]
        nocc = occidx.size
        nvir = viridx.size
        nov = nocc * nvir

        # PySCF's restricted get_ab constructs the singlet matrix only.
        a_s = rhf.get_ab(mf, frozen=self.frozen)[0].reshape(nov, nov)
        a_t = _get_triplet_a(mf, mask).reshape(nov, nov)

        self.soc_ao = get_ao_soc(mf, self.soctype)
        self.soc_mo = lib.einsum(
            'up,xuv,vq->xpq', mo_coeff.conj(), self.soc_ao, mo_coeff, optimize=True,
        )
        z_minus = 2 * self.soc_mo[0]  # zx - i zy
        z_z = np.sqrt(2) * self.soc_mo[1]  # zz
        z_plus = -2 * self.soc_mo[2]  # zx + i zy

        eye_occ = np.eye(nocc)
        eye_vir = np.eye(nvir)

        def one_body_matrix(z, hole_sign):
            z_oo = z[np.ix_(occidx, occidx)]
            z_vv = z[np.ix_(viridx, viridx)]
            matrix = lib.einsum('ij,ab->iajb', eye_occ, z_vv)
            matrix += hole_sign * lib.einsum('ab,ji->iajb', eye_vir, z_oo)
            return matrix.reshape(nov, nov)

        d_minus = one_body_matrix(z_minus, -1)
        d_z = one_body_matrix(z_z, -1)
        d_plus = one_body_matrix(z_plus, -1)
        q_minus = one_body_matrix(z_minus, 1)
        q_z = one_body_matrix(z_z, 1)
        q_plus = one_body_matrix(z_plus, 1)

        offset = int(self.include_reference)
        hamiltonian = np.zeros((offset + 4 * nov, offset + 4 * nov), dtype=np.complex128)
        block = [slice(offset + k * nov, offset + (k + 1) * nov) for k in range(4)]  #  [ref, S, T-1, T0, T+1]
        hamiltonian[block[0], block[0]] = a_s
        hamiltonian[block[1], block[1]] = a_t - q_z / 2
        hamiltonian[block[2], block[2]] = a_t
        hamiltonian[block[3], block[3]] = a_t + q_z / 2

        hamiltonian[block[0], block[1]] = d_minus / np.sqrt(8)
        hamiltonian[block[1], block[0]] = d_plus / np.sqrt(8)
        hamiltonian[block[0], block[2]] = d_z / 2
        hamiltonian[block[2], block[0]] = d_z / 2
        hamiltonian[block[0], block[3]] = -d_plus / np.sqrt(8)
        hamiltonian[block[3], block[0]] = -d_minus / np.sqrt(8)
        hamiltonian[block[1], block[2]] = q_plus / np.sqrt(8)
        hamiltonian[block[2], block[1]] = q_minus / np.sqrt(8)
        hamiltonian[block[2], block[3]] = q_plus / np.sqrt(8)
        hamiltonian[block[3], block[2]] = q_minus / np.sqrt(8)

        if self.include_reference:
            z_minus_vo = z_minus[np.ix_(viridx, occidx)].T.reshape(-1)
            z_z_vo = z_z[np.ix_(viridx, occidx)].T.reshape(-1)
            z_plus_vo = z_plus[np.ix_(viridx, occidx)].T.reshape(-1)
            reference_coupling = np.concatenate((
                np.zeros(nov, dtype=np.complex128),
                0.5 * z_plus_vo,
                z_z_vo / np.sqrt(2),
                -0.5 * z_minus_vo,
            ))
            hamiltonian[1:, 0] = reference_coupling
            hamiltonian[0, 1:] = reference_coupling.conj()
        return hamiltonian

    def get_init_guess(self, hdiag, nstates=None):
        """Return Koopmans guesses, retaining all states at the cutoff degeneracy."""
        if nstates is None:
            nstates = self.nstates
        nstates = min(nstates, hdiag.size)
        if nstates <= 0:
            raise ValueError('nstates must be positive')
        nguess = min(nstates + 3, hdiag.size)
        order = np.argsort(hdiag.real)
        threshold = hdiag[order[nguess - 1]].real + self.deg_eia_thresh
        idx = order[hdiag[order].real <= threshold]
        x0 = np.zeros((idx.size, hdiag.size), dtype=np.complex128)
        x0[np.arange(idx.size), idx] = 1
        return x0

    def kernel(self, x0=None, nstates=None):
        """Diagonalize the complex Hermitian SOC-TDA matrix."""
        cpu0 = (logger.process_clock(), logger.perf_counter())
        self.check_sanity()
        self.dump_flags()
        if nstates is None:
            nstates = self.nstates
        else:
            self.nstates = nstates

        log = logger.Logger(self.stdout, self.verbose)
        vind, hdiag = self.gen_vind()
        dimension = hdiag.size
        nstates = min(nstates, dimension)
        precond = self.get_precond(hdiag)

        def all_eigs(w, v, nroots, envs):
            return w, v, np.arange(w.size)

        if x0 is None:
            x0 = self.get_init_guess(hdiag, nstates)
        self.converged, self.e, x1 = lr_eigh(
            vind,
            x0,
            precond,
            tol_residual=self.conv_tol,
            lindep=self.lindep,
            nroots=nstates,
            x0sym=None,
            pick=all_eigs,
            max_cycle=self.max_cycle,
            max_memory=self.max_memory,
            verbose=log,
        )
        self.xy = [(xi, 0) for xi in x1]

        if self.chkfile:
            lib.chkfile.save(self.chkfile, 'tddft/e', self.e)
            lib.chkfile.save(self.chkfile, 'tddft/xy', self.xy)

        log.timer('SOTDDFT', *cpu0)
        self._finalize()
        return self.e, self.xy


scf.hf.RHF.SOTDDFT = lib.class_as_method(SOTDDFT)


__all__ = ['SOTDDFT']
