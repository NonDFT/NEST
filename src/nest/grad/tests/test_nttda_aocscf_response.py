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

"""Orbital-response checks for average-occupation AOCSCF gradients."""

import unittest

import numpy as np
from scipy.linalg import expm


from pyscf import dft, gto, lib


from nest.grad.nttda.ensemble import (  # noqa: E402
    make_hessian_transpose_action,
    pack_m_matrix,
    zvector_adjoint_matrix,
    zvector_probe_densities,
)
from nest.grad.nttda.delta_s_zero import (  # noqa: E402
    grad_elec,
    same_spin_ledger_scalar,
)
from nest.grad.nttda.delta_s_minus_one import (  # noqa: E402
    grad_elec as spin_lowering_grad_elec,
    spin_lowering_ledger_scalar,
)
from nest import aocscf  # noqa: E402
from nest.nttda import NTTDA  # noqa: E402


class AOCSCFOrbitalResponse(unittest.TestCase):
    @staticmethod
    def make_reference():
        mol = gto.M(
            atom="Li 0 0 0; H 0 0 3.0",
            basis="sto-3g",
            charge=1,
            spin=1,
            unit="Bohr",
            verbose=0,
        )
        mf = mol.ROKS(xc="SVWN").average_occ().set(
            conv_tol=1e-13,
            conv_tol_grad=1e-10,
            max_cycle=100,
            verbose=0,
        )
        mf.grids.level = 0
        mf.kernel()
        if not mf.converged:
            raise RuntimeError("AOCSCF reference did not converge")
        return mf

    @staticmethod
    def make_spin_one_reference():
        mol = gto.M(
            atom="C 0 0 0; H 0 0 2.0; H 0 1.7 -0.5",
            basis="sto-3g",
            spin=2,
            unit="Bohr",
            verbose=0,
        )
        mf = mol.ROKS(xc="SVWN").average_occ().set(
            conv_tol=1e-12,
            conv_tol_grad=1e-9,
            max_cycle=150,
            verbose=0,
        )
        mf.grids.level = 0
        mf.kernel()
        if not mf.converged:
            raise RuntimeError("spin-one AOCSCF reference did not converge")
        return mf

    @staticmethod
    def _common_fock(mf, mo_coeff, mo_occ):
        density = mf.make_rdm1(mo_coeff, mo_occ)
        veff = mf.get_veff(mf.mol, density)
        return mf.get_hcore() + veff[0]

    def test_charge_response_tracks_a_reused_reference(self):
        mol_a = gto.M(
            atom="Li 0 0 0; H 0 0 3.0",
            basis="sto-3g",
            charge=1,
            spin=1,
            unit="Bohr",
            verbose=0,
        )
        mf = mol_a.ROKS(xc="SVWN").average_occ().set(
            conv_tol=1e-12,
            conv_tol_grad=1e-9,
            max_cycle=100,
            verbose=0,
        )
        mf.grids.level = 0
        mf.kernel()
        rng = np.random.default_rng(7)
        density = rng.standard_normal((mol_a.nao_nr(),) * 2)
        density = 0.5 * (density + density.T)
        response_a = lib.view(mf, dft.rks.RKS).gen_response(hermi=1)(density)

        mol_b = gto.M(
            atom="Li 0 0 0; H 0 0 3.2",
            basis="sto-3g",
            charge=1,
            spin=1,
            unit="Bohr",
            verbose=0,
        )
        mf.reset(mol_b)
        mf.grids.level = 0
        mf.kernel()
        response_b = lib.view(mf, dft.rks.RKS).gen_response(hermi=1)(density)

        fresh = mol_b.ROKS(xc="SVWN").average_occ().set(
            conv_tol=1e-12,
            conv_tol_grad=1e-9,
            max_cycle=100,
            verbose=0,
        )
        fresh.grids.level = 0
        fresh.kernel()
        response_fresh = lib.view(fresh, dft.rks.RKS).gen_response(hermi=1)(density)

        np.testing.assert_allclose(
            response_b, response_fresh, atol=1e-8, rtol=0,
        )
        self.assertGreater(np.max(np.abs(response_b - response_a)), 1e-4)

    def test_hessian_action_matches_orbital_rotation_finite_difference(self):
        mf = self.make_reference()
        tdobj = NTTDA(mf)
        action, pairs = make_hessian_transpose_action(tdobj)
        rng = np.random.default_rng(19)
        vector = rng.normal(size=len(pairs))

        mo = np.asarray(mf.mo_coeff)
        occ = np.asarray(mf.mo_occ)
        kappa = np.zeros((mo.shape[1], mo.shape[1]))
        for value, (p, q, _name) in zip(vector, pairs):
            kappa[p, q] = value
            kappa[q, p] = -value

        step = 1e-5
        gradients = []
        for sign in (1.0, -1.0):
            displaced_mo = mo @ expm(sign * step * kappa)
            fock = self._common_fock(mf, displaced_mo, occ)
            gradients.append(mf.get_grad(displaced_mo, occ, fock))
        finite_difference = (gradients[0] - gradients[1]) / (2.0 * step)

        np.testing.assert_allclose(
            action(vector), finite_difference, atol=1e-8, rtol=0,
        )

    def test_adjoint_and_probe_are_consistent_with_explicit_hessian(self):
        mf = self.make_reference()
        tdobj = NTTDA(mf)
        action, pairs = make_hessian_transpose_action(tdobj)
        identity = np.eye(len(pairs))
        hessian = np.asarray(action(identity)).T
        rng = np.random.default_rng(23)
        zvector = rng.normal(size=len(pairs))

        adjoint = zvector_adjoint_matrix(tdobj, pairs, zvector)
        np.testing.assert_allclose(
            pack_m_matrix(adjoint, pairs),
            hessian.T @ zvector,
            atol=1e-10,
            rtol=0,
        )

        probe_alpha, probe_beta = zvector_probe_densities(
            tdobj, pairs, zvector,
        )
        perturbation = rng.normal(size=(mf.mol.nao_nr(),) * 2)
        perturbation = perturbation + perturbation.T
        mo = np.asarray(mf.mo_coeff)
        occ = np.asarray(mf.mo_occ)
        fock_mo = mo.T @ perturbation @ mo
        expected = sum(
            value * (occ[q] - occ[p]) * fock_mo[p, q]
            for value, (p, q, _name) in zip(zvector, pairs)
        )
        actual = np.einsum(
            "ij,ji", probe_alpha + probe_beta, perturbation,
        )
        self.assertAlmostEqual(actual, expected, places=11)

    def test_same_spin_m_matrix_is_the_orbital_derivative(self):
        mf = self.make_reference()
        tdobj = NTTDA(mf).set(
            deltaS=0,
            nstates=2,
            conv_tol=1e-10,
            max_cycle=200,
            verbose=0,
        ).run()
        xy = tdobj.xy[1]
        result = grad_elec(
            mf.nuc_grad_method(), tdobj, xy, atmlst=(),
        )

        rng = np.random.default_rng(29)
        perturbation = rng.normal(size=(mf.mo_coeff.shape[1],) * 2)
        original = np.array(mf.mo_coeff, copy=True)
        step = 1e-6
        values = []
        try:
            for sign in (1.0, -1.0):
                mf.mo_coeff = original @ (
                    np.eye(original.shape[1])
                    + sign * step * perturbation
                )
                values.append(same_spin_ledger_scalar(tdobj, xy))
        finally:
            mf.mo_coeff = original
        finite_difference = (values[0] - values[1]) / (2.0 * step)
        analytic = np.trace(result.m_matrix.T @ perturbation)
        self.assertAlmostEqual(analytic, finite_difference, places=8)

    def test_spin_lowering_m_matrix_is_the_orbital_derivative(self):
        mf = self.make_spin_one_reference()
        tdobj = NTTDA(mf).set(
            deltaS=-1,
            nstates=2,
            conv_tol=1e-9,
            max_cycle=200,
            verbose=0,
        ).run()
        xy = tdobj.xy[0]
        result = spin_lowering_grad_elec(
            mf.nuc_grad_method(), tdobj, xy, atmlst=(),
        )

        rng = np.random.default_rng(31)
        perturbation = rng.normal(size=(mf.mo_coeff.shape[1],) * 2)
        original = np.array(mf.mo_coeff, copy=True)
        step = 1e-6
        values = []
        try:
            for sign in (1.0, -1.0):
                mf.mo_coeff = original @ (
                    np.eye(original.shape[1]) + sign * step * perturbation
                )
                values.append(spin_lowering_ledger_scalar(tdobj, xy))
        finally:
            mf.mo_coeff = original
        finite_difference = (values[0] - values[1]) / (2.0 * step)
        analytic = np.trace(result.m_matrix.T @ perturbation)
        self.assertAlmostEqual(analytic, finite_difference, places=8)


if __name__ == "__main__":
    unittest.main()
