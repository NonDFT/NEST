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

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
from scipy.sparse.linalg import LinearOperator, gmres

from pyscf import dft, lib
from pyscf.grad import rhf as rhf_grad
from pyscf.hessian import rhf as rhf_hess
from pyscf.hessian import rks as rks_hess
from pyscf.lib import logger
from pyscf.scf import _response_functions


_OCC_TOL = 1e-8


@dataclass(frozen=True)
class RotationSpace:
    """Packed nonredundant rotations and their orbital occupations."""

    p: np.ndarray
    q: np.ndarray
    f: np.ndarray
    nalpha: np.ndarray
    nbeta: np.ndarray

    @property
    def size(self) -> int:
        return int(self.p.size)

    @property
    def occupation_gap(self) -> np.ndarray:
        return self.f[self.q] - self.f[self.p]

    def unpack(self, vector: np.ndarray) -> np.ndarray:
        """Map a packed vector to the full real anti-symmetric MO matrix."""
        vector = np.asarray(vector, dtype=float)
        if vector.shape != (self.size,):
            raise ValueError(
                f"Expected a packed rotation of shape {(self.size,)}, "
                f"got {vector.shape}."
            )
        matrix = np.zeros((self.f.size, self.f.size))
        matrix[self.p, self.q] = vector
        matrix[self.q, self.p] = -vector
        return matrix

    def pack(self, matrix: np.ndarray) -> np.ndarray:
        """Extract entries matching the packed (p,q) rotation order."""
        return np.asarray(matrix)[..., self.p, self.q]


def _block_pairs(rows: np.ndarray, cols: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pairs matching ``matrix[np.ix_(rows, cols)].ravel()`` order."""
    rows = np.asarray(rows, dtype=int)
    cols = np.asarray(cols, dtype=int)
    if rows.size == 0 or cols.size == 0:
        empty = np.empty(0, dtype=int)
        return empty, empty
    return np.repeat(rows, cols.size), np.tile(cols, rows.size)


def _rotation_space(mo_occ: np.ndarray) -> RotationSpace:
    f = np.asarray(mo_occ, dtype=float)
    if f.ndim != 1:
        raise ValueError(f"Spin-averaged occupations must be a one-dimensional array; got {f.shape}.")

    is_c = np.isclose(f, 2.0, atol=_OCC_TOL, rtol=0.0)
    is_o = np.isclose(f, 1.0, atol=_OCC_TOL, rtol=0.0)
    is_v = np.isclose(f, 0.0, atol=_OCC_TOL, rtol=0.0)
    if not np.all(is_c | is_o | is_v):
        bad = np.where(~(is_c | is_o | is_v))[0]
        raise NotImplementedError(
            "This implementation requires occupations 0, 1, or 2. "
            f"Nonstandard occupations were found at MO indices {bad.tolist()}."
        )

    c = np.where(is_c)[0]
    o = np.where(is_o)[0]
    v = np.where(is_v)[0]
    blocks = (_block_pairs(o, c), _block_pairs(v, c), _block_pairs(v, o))
    p = np.concatenate([block[0] for block in blocks])
    q = np.concatenate([block[1] for block in blocks])

    nalpha = (f > 0.0).astype(float)
    nbeta = np.isclose(f, 2.0, atol=_OCC_TOL, rtol=0.0).astype(float)
    return RotationSpace(p=p, q=q, f=f, nalpha=nalpha, nbeta=nbeta)


def _ao_density(mo_coeff: np.ndarray, occupation: np.ndarray) -> np.ndarray:
    """Return D_AO = C occupation C^dagger."""
    return (mo_coeff * occupation) @ mo_coeff.conj().T


def _transform_ao_to_mo(mo_coeff: np.ndarray, matrices: np.ndarray) -> np.ndarray:
    """Transform one AO matrix or a leading batch of AO matrices to the MO basis."""
    return np.einsum("up,...uv,vq->...pq", mo_coeff.conj(), matrices, mo_coeff, optimize=True)


def _full_overlap_derivative(
    one_sided_s1: np.ndarray,
    atom: int,
    aoslices: np.ndarray,
) -> np.ndarray:
    """Build S^A from PySCF's one-sided overlap derivative matrix."""
    p0, p1 = aoslices[atom][2:]
    s1 = np.zeros_like(one_sided_s1)
    s1[:, p0:p1, :] += one_sided_s1[:, p0:p1, :]
    s1[:, :, p0:p1] += one_sided_s1[:, p0:p1, :].transpose(0, 2, 1)
    return s1


def _fractional_rks_fock_skeleton(
    charge_mf,
    mo_coeff: np.ndarray,
    mo_occ: np.ndarray,
) -> np.ndarray:
    """Return F_AO^(0,[A]) for all atoms and Cartesian components.

    This is PySCF's RKS ``make_h1`` construction with the density corrected
    from ``2 C_occ C_occ^T`` to the fractional-occupation density
    ``C mo_occ C^T`` required by spin-averaged SCF.  Range-separated exchange is included
    through PySCF's ``(omega, alpha, hyb)`` decomposition.
    """
    mol = charge_mf.mol
    dm0 = _ao_density(mo_coeff, mo_occ)
    hessobj = rks_hess.Hessian(charge_mf)
    hcore_deriv = charge_mf.nuc_grad_method().hcore_generator(mol)

    ni = charge_mf._numint
    ni.libxc.test_deriv_order(charge_mf.xc, 2, raise_error=True)
    omega, alpha, hyb = ni.rsh_and_hybrid_coeff(charge_mf.xc, spin=mol.spin)
    hybrid = ni.libxc.is_hybrid_xc(charge_mf.xc)

    memory_now = lib.current_memory()[0]
    max_memory = max(2000, charge_mf.max_memory * 0.9 - memory_now)
    h1ao = rks_hess._get_vxc_deriv1(hessobj, mo_coeff, mo_occ, max_memory)

    aoslices = mol.aoslice_by_atom()
    for atom, (shl0, shl1, p0, p1) in enumerate(aoslices):
        shls_slice = (shl0, shl1) + (0, mol.nbas) * 3
        if hybrid:
            vj1, vj2, vk1, vk2 = rhf_hess._get_jk(
                mol,
                "int2e_ip1",
                3,
                "s2kl",
                [
                    "ji->s2kl",
                    -dm0[:, p0:p1],
                    "lk->s1ij",
                    -dm0,
                    "li->s1kj",
                    -dm0[:, p0:p1],
                    "jk->s1il",
                    -dm0,
                ],
                shls_slice=shls_slice,
            )
            veff = vj1 - 0.5 * hyb * vk1
            veff[:, p0:p1] += vj2 - 0.5 * hyb * vk2
            if omega != 0.0:
                with mol.with_range_coulomb(omega):
                    vk1, vk2 = rhf_hess._get_jk(
                        mol,
                        "int2e_ip1",
                        3,
                        "s2kl",
                        [
                            "li->s1kj",
                            -dm0[:, p0:p1],
                            "jk->s1il",
                            -dm0,
                        ],
                        shls_slice=shls_slice,
                    )
                veff -= 0.5 * (alpha - hyb) * vk1
                veff[:, p0:p1] -= 0.5 * (alpha - hyb) * vk2
        else:
            vj1, vj2 = rhf_hess._get_jk(
                mol,
                "int2e_ip1",
                3,
                "s2kl",
                ["ji->s2kl", -dm0[:, p0:p1], "lk->s1ij", -dm0],
                shls_slice=shls_slice,
            )
            veff = vj1
            veff[:, p0:p1] += vj2

        h1ao[atom] += veff + veff.transpose(0, 2, 1)
        h1ao[atom] += hcore_deriv(atom)

    return h1ao


class AverageOccupationGradients(lib.StreamObject):
    """Analytic gradient of the high-spin energy at spin-averaged SCF orbitals."""

    _keys = {
        "base",
        "mol",
        "max_memory",
        "conv_tol",
        "max_cycle",
        "restart",
        "grid_response",
        "atmlst",
        "de",
        "z",
        "g_hs",
        "g_avg_occ",
        "b",
        "e_hs_unrelaxed",
    }

    def __init__(self, mf):
        self.base = mf
        self.mol = mf.mol
        self.verbose = mf.verbose
        self.stdout = mf.stdout
        self.max_memory = mf.max_memory

        self.conv_tol = 1e-9
        self.max_cycle = 80
        self.restart = 40
        self.grid_response = False
        self.atmlst = None

        self.de = None
        self.z = None
        self.g_hs = None
        self.g_avg_occ = None
        self.b = None
        self.e_hs_unrelaxed = None

        self._space = None
        self._charge_mf = None
        self._hs_mf = None
        self._charge_response = None
        self._c0 = None
        self._f0ao = None
        self._f0mo = None
        self._f_hs_ao = None
        self._f_hs_mo = None
        self._dm_hs = None
        self._w_hs_mo = None

    def dump_flags(self, verbose=None):
        log = logger.new_logger(self, verbose)
        log.info("******** Spin-averaged SCF high-spin analytic gradient ********")
        log.info("Z-vector tolerance = %.3g", self.conv_tol)
        log.info("Z-vector max cycles = %d", self.max_cycle)
        log.info("GMRES restart = %d", self.restart)
        log.info("grid response = %s", self.grid_response)
        return self

    def _validate(self) -> None:
        mf = self.base
        mol = self.mol
        if getattr(mf, "mo_coeff", None) is None or getattr(mf, "mo_occ", None) is None:
            raise RuntimeError("Run spin-averaged SCF before requesting its analytic gradient.")
        if hasattr(mf, "converged") and not mf.converged:
            warnings.warn("Spin-averaged SCF is not converged; its analytic gradient is not stationary.")
        if np.iscomplexobj(mf.mo_coeff) and np.max(np.abs(np.asarray(mf.mo_coeff).imag)) > 1e-12:
            raise NotImplementedError("Complex-orbital spin-averaged gradients are not implemented.")
        if self.grid_response:
            raise NotImplementedError(
                "Moving-grid response is not implemented consistently in B^(0,A); "
                "use grid_response=False."
            )
        if getattr(mf, "with_df", None) is not None:
            raise NotImplementedError("Density-fitted spin-averaged gradients are not implemented.")
        if getattr(mf, "with_x2c", None) is not None:
            raise NotImplementedError("X2C spin-averaged gradients are not implemented.")
        if getattr(mf, "with_solvent", None) is not None:
            raise NotImplementedError("Solvent-response spin-averaged gradients are not implemented.")
        if hasattr(mf, "do_nlc") and mf.do_nlc():
            raise NotImplementedError("Nonlocal-correlation (NLC/VV10) spin-averaged gradients are not implemented.")
        if hasattr(mf, "do_disp") and mf.do_disp():
            raise NotImplementedError("Dispersion-corrected spin-averaged gradients are not implemented.")
        if getattr(mol, "dimension", 3) != 3:
            raise NotImplementedError("Only molecular (three-dimensional) calculations are supported.")

    def _build_intermediates(self) -> None:
        mf = self.base
        mol = self.mol
        c0 = np.asarray(mf.mo_coeff).real
        space = _rotation_space(np.asarray(mf.mo_occ))

        # The factory dft.RKS returns ROKS when mol.spin != 0.  A view keeps
        # the reference's numerical settings but selects the RKS charge kernel.
        charge_mf = lib.view(mf, dft.rks.RKS)
        charge_mf.mo_coeff = c0
        charge_mf.mo_occ = space.f

        dm0 = _ao_density(c0, space.f)
        hcore = charge_mf.get_hcore(mol)
        f0ao = hcore + charge_mf.get_veff(mol, dm0)
        f0mo = _transform_ao_to_mo(c0, f0ao)

        charge_response = _response_functions._gen_rhf_response(
            charge_mf,
            mo_coeff=c0,
            mo_occ=space.f,
            singlet=None,
            hermi=1,
            max_memory=self.max_memory,
            with_nlc=False,
        )

        hs_mf = lib.view(mf, dft.roks.ROKS)
        hs_mf.mo_coeff = c0
        hs_mf.mo_occ = space.f
        dm_hs = hs_mf.make_rdm1(c0, space.f)
        veff_hs = hs_mf.get_veff(mol, dm_hs)
        f_hs_ao = np.asarray((hcore + veff_hs[0], hcore + veff_hs[1]))
        f_hs_mo = _transform_ao_to_mo(c0, f_hs_ao)

        occ_spin = np.asarray((space.nalpha, space.nbeta))
        w_hs_mo = 0.5 * np.sum(
            occ_spin[:, :, None] * f_hs_mo + f_hs_mo * occ_spin[:, None, :],
            axis=0,
        )

        self._space = space
        self._charge_mf = charge_mf
        self._hs_mf = hs_mf
        self._charge_response = charge_response
        self._c0 = c0
        self._f0ao = f0ao
        self._f0mo = f0mo
        self._f_hs_ao = f_hs_ao
        self._f_hs_mo = f_hs_mo
        self._dm_hs = dm_hs
        self._w_hs_mo = w_hs_mo

        gap = space.occupation_gap
        self.g_avg_occ = 2.0 * gap * space.pack(f0mo)
        self.g_hs = 2.0 * (
            (space.nalpha[space.q] - space.nalpha[space.p]) * space.pack(f_hs_mo[0])
            + (space.nbeta[space.q] - space.nbeta[space.p]) * space.pack(f_hs_mo[1])
        )

    def hessian_vector_product(self, vector: np.ndarray) -> np.ndarray:
        """Evaluate A^(0) vector without constructing the orbital Hessian."""
        if self._space is None:
            self._validate()
            self._build_intermediates()

        space = self._space
        kappa = space.unpack(vector)
        delta_dm_mo = kappa * space.f[None, :] - space.f[:, None] * kappa
        delta_dm_ao = self._c0 @ delta_dm_mo @ self._c0.T
        delta_f_ao = self._charge_response(delta_dm_ao)
        delta_f_mo = _transform_ao_to_mo(self._c0, delta_f_ao)
        moving_mo = self._f0mo @ kappa - kappa @ self._f0mo
        result = 2.0 * space.occupation_gap * space.pack(moving_mo + delta_f_mo)
        return np.asarray(result).real

    def _solve_z(self) -> np.ndarray:
        space = self._space
        if space.size == 0:
            return np.empty(0)

        operator = LinearOperator(
            (space.size, space.size),
            matvec=self.hessian_vector_product,
            rmatvec=self.hessian_vector_product,
            dtype=float,
        )

        # The exact real-orbital spin-averaged Hessian is symmetric, so A^T z = g_HS
        # is solved with the same matrix-free action.  This diagonal contains
        # the one-electron commutator part and is used only as a preconditioner.
        diagonal = 2.0 * space.occupation_gap * (
            self._f0mo.diagonal()[space.p] - self._f0mo.diagonal()[space.q]
        )
        floor = max(1e-8, 1e-6 * np.max(np.abs(diagonal)))
        safe_diagonal = np.where(
            np.abs(diagonal) > floor,
            diagonal,
            np.where(diagonal < 0.0, -floor, floor),
        )
        preconditioner = LinearOperator(
            operator.shape,
            matvec=lambda x: np.asarray(x) / safe_diagonal,
            dtype=float,
        )

        residuals = []
        common = dict(
            A=operator,
            b=np.asarray(self.g_hs).real,
            M=preconditioner,
            restart=self.restart,
            maxiter=self.max_cycle,
            callback=residuals.append,
        )
        try:
            z, info = gmres(rtol=self.conv_tol, atol=0.0, callback_type="pr_norm", **common)
        except TypeError:  # SciPy < 1.12 compatibility
            z, info = gmres(tol=self.conv_tol, atol=0.0, **common)

        if info != 0:
            last = residuals[-1] if residuals else np.nan
            raise RuntimeError(
                "Spin-averaged Z-vector GMRES did not converge: "
                f"info={info}, last preconditioned residual={last:.3e}. "
                "Increase max_cycle/restart or inspect a near-singular orbital Hessian."
            )
        return z

    def _high_spin_unrelaxed_gradient(self) -> np.ndarray:
        """Evaluate E_HS^(A), including the nonstationary overlap term."""
        mol = self.mol
        c0 = self._c0
        dm_hs = self._dm_hs
        hs_grad = self._hs_mf.nuc_grad_method()
        hs_grad.grid_response = False
        hs_grad.max_memory = self.max_memory

        hcore_deriv = hs_grad.hcore_generator(mol)
        one_sided_s1 = hs_grad.get_ovlp(mol)

        dm_hs_for_gradient = hs_grad._tag_rdm1(
            np.asarray(dm_hs), c0, self._space.f
        )
        vhf = hs_grad.get_veff(mol, dm_hs_for_gradient)
        dm_total = dm_hs[0] + dm_hs[1]
        w_hs_ao = c0 @ self._w_hs_mo @ c0.T

        if mol._pseudo:
            from pyscf.gto.pp_int import vpploc_nuc_grad, vppnl_nuc_grad

            de = vpploc_nuc_grad(mol, dm_total)
            de += vppnl_nuc_grad(mol, dm_total)
        else:
            de = np.zeros((mol.natm, 3))

        aoslices = mol.aoslice_by_atom()
        for atom, (_, _, p0, p1) in enumerate(aoslices):
            de[atom] += np.einsum("xij,ij->x", hcore_deriv(atom), dm_total)
            de[atom] += 2.0 * np.einsum(
                "sxij,sij->x", vhf[:, :, p0:p1], dm_hs[:, p0:p1]
            )
            # -Tr[W_HS^MO S_MO^A], written with PySCF's one-sided S derivative.
            de[atom] -= 2.0 * np.einsum(
                "xij,ij->x", one_sided_s1[:, p0:p1], w_hs_ao[p0:p1]
            )

        return de + hs_grad.grad_nuc(mol)

    def _build_b(self) -> np.ndarray:
        """Build B_i^(0,A) for all atoms and Cartesian components."""
        mol = self.mol
        c0 = self._c0
        space = self._space
        aoslices = mol.aoslice_by_atom()
        one_sided_s1 = rhf_grad.get_ovlp(mol)
        fock_skeleton = _fractional_rks_fock_skeleton(
            self._charge_mf, c0, space.f
        )

        b = np.empty((space.size, mol.natm, 3))
        for atom in range(mol.natm):
            s1ao = _full_overlap_derivative(one_sided_s1, atom, aoslices)
            s1mo = _transform_ao_to_mo(c0, s1ao)

            anticommutator_sf = s1mo * space.f[None, None, :]
            anticommutator_sf += space.f[None, :, None] * s1mo
            delta_dm_sym_mo = -0.5 * anticommutator_sf
            delta_dm_sym_ao = np.einsum(
                "up,xpq,vq->xuv", c0, delta_dm_sym_mo, c0.conj(), optimize=True
            )
            response_ao = self._charge_response(delta_dm_sym_ao)

            explicit_mo = _transform_ao_to_mo(c0, fock_skeleton[atom])
            response_mo = _transform_ao_to_mo(c0, response_ao)
            overlap_mo = -0.5 * (
                np.einsum("xpq,qr->xpr", s1mo, self._f0mo, optimize=True)
                + np.einsum("pq,xqr->xpr", self._f0mo, s1mo, optimize=True)
            )
            fock_fixed_k = explicit_mo + response_mo + overlap_mo
            b[:, atom, :] = (
                2.0
                * space.occupation_gap[:, None]
                * space.pack(fock_fixed_k).T
            )
        return b.real

    def kernel(self, atmlst=None, verbose=None) -> np.ndarray:
        """Compute the high-spin nuclear gradient at spin-averaged orbitals."""
        log = logger.new_logger(self, verbose)
        self._validate()
        self.dump_flags(verbose)
        self._build_intermediates()

        max_g0 = float(np.max(np.abs(self.g_avg_occ))) if self.g_avg_occ.size else 0.0
        log.info("max |g_avg_occ| = %.6g", max_g0)
        scf_grad_tol = getattr(self.base, "conv_tol_grad", 0.0) or 0.0
        if max_g0 > max(1e-6, 100.0 * scf_grad_tol):
            warnings.warn(
                f"The packed spin-averaged orbital gradient is not small (max={max_g0:.3e}); "
                "the analytic-gradient stationarity equation may be inaccurate."
            )

        self.z = self._solve_z()
        self.e_hs_unrelaxed = self._high_spin_unrelaxed_gradient()
        self.b = self._build_b()
        z_correction = np.einsum("i,iax->ax", self.z, self.b, optimize=True)
        de = self.e_hs_unrelaxed - z_correction

        if self.mol.symmetry:
            de = rhf_grad.symmetrize(self.mol, de)

        self.atmlst = atmlst
        result = de if atmlst is None else de[np.asarray(atmlst, dtype=int)]
        self.de = result

        if log.verbose >= logger.NOTE:
            logger.note(self, "------------ Spin-averaged reference gradients ------------")
            rhf_grad._write(log, self.mol, result, atmlst)
            logger.note(self, "----------------------------------------------------------")
        return result

    grad = kernel


Gradients = AverageOccupationGradients


__all__ = ["AverageOccupationGradients", "Gradients"]
