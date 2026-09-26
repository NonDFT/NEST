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

"""Finite-difference NTTDA gradients on an AOCSCF reference."""

import unittest

import numpy as np

from pyscf import gto
from nest import aocscf
from nest.nttda import NTTDA


class AOCSCFFiniteDifferenceGradient(unittest.TestCase):
    @staticmethod
    def make_td():
        mol = gto.M(
            atom="Li 0 0 0; H 0 0 3.0",
            basis="sto-3g",
            charge=1,
            spin=1,
            unit="Bohr",
            verbose=0,
        )
        mf = mol.ROKS(xc="SVWN").average_occ().set(
            conv_tol=1e-12,
            conv_tol_grad=1e-9,
            max_cycle=100,
            verbose=0,
        )
        mf.grids.level = 0
        mf.kernel()
        if not mf.converged:
            raise RuntimeError("AOCSCF test reference did not converge")
        tdobj = NTTDA(mf).set(
            deltaS=0,
            nstates=2,
            conv_tol=1e-8,
            lindep=1e-18,
            max_cycle=100,
            verbose=0,
        ).run()
        return tdobj

    def test_finite_difference_total_energy_gradient(self):
        tdobj = self.make_td()
        gradient = tdobj.Gradients().set(
            verbose=0,
            fixed_grid=False,
            root_overlap_tol=0.5,
        )
        self.assertTrue(tdobj.Gradients().fixed_grid)
        from nest.grad.nttda import _displaced_reference

        displaced = _displaced_reference(tdobj._scf, tdobj.mol.copy(), False)
        self.assertIsInstance(displaced, type(tdobj._scf))
        result = gradient.kernel(
            state=2,
            method="finite_diff",
            step=2e-3,
        )
        self.assertEqual(result.shape, (2, 3))
        self.assertTrue(np.all(np.isfinite(result)))
        np.testing.assert_allclose(
            result,
            [[0, 0, 0.091524381309104896],
             [0, 0, -0.091524381308882852]],
            atol=2e-5,
            rtol=0,
        )
        np.testing.assert_allclose(result.sum(axis=0), 0, atol=2e-5, rtol=0)
        np.testing.assert_allclose(result[:, :2], 0, atol=2e-5, rtol=0)
        self.assertGreater(abs(result[0, 2]), 1e-3)

        gradient.fixed_grid = True
        analytic = gradient.kernel(state=2, method="analytic")
        fixed_grid_difference = gradient.kernel(
            state=2,
            method="finite_diff",
            step=2e-3,
        )
        np.testing.assert_allclose(
            analytic, fixed_grid_difference, atol=2e-5, rtol=0,
        )

    def test_all_spin_channels_have_a_finite_difference_path(self):
        mol = gto.M(
            atom="C 0 0 0; H 0 0 2.0; H 0 1.7 -0.5",
            basis="sto-3g",
            spin=2,
            unit="Bohr",
            verbose=0,
        )
        mf = mol.ROKS(xc="SVWN").average_occ().set(
            conv_tol=1e-11,
            max_cycle=150,
            verbose=0,
        )
        mf.grids.level = 0
        mf.kernel()
        self.assertTrue(mf.converged)

        for delta_s in (-1, 0, 1):
            with self.subTest(deltaS=delta_s):
                tdobj = NTTDA(mf).set(
                    deltaS=delta_s,
                    nstates=2,
                    conv_tol=1e-6,
                    max_cycle=200,
                    verbose=0,
                ).run()
                result = tdobj.Gradients().set(
                    verbose=0,
                    root_overlap_tol=0.5,
                ).kernel(
                    state=1,
                    atmlst=[0],
                    method="finite_diff",
                    step=2e-3,
                )
                self.assertEqual(result.shape, (1, 3))
                self.assertTrue(np.all(np.isfinite(result)))

    def test_spin_lowering_analytic_matches_fixed_grid_finite_difference(self):
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
        self.assertTrue(mf.converged)
        tdobj = NTTDA(mf).set(
            deltaS=-1,
            nstates=2,
            conv_tol=1e-9,
            max_cycle=200,
            verbose=0,
        ).run()
        gradient = tdobj.Gradients().set(
            verbose=0,
            fixed_grid=True,
            root_overlap_tol=0.5,
        )
        analytic = gradient.kernel(state=1, atmlst=[1], method="analytic")
        finite_difference = gradient.kernel(
            state=1,
            atmlst=[1],
            method="finite_diff",
            step=2e-3,
        )
        np.testing.assert_allclose(
            analytic, finite_difference, atol=3e-5, rtol=0,
        )

    def test_representative_functional_families(self):
        cases = (
            (0, "HF", False),
            (0, "PBE", False),
            (0, "M06-2X", False),
            (0, "CAM-B3LYP", False),
            (-1, "PBE", False),
            (-1, "M06-2X", True),
        )
        for delta_s, xc, nobeta in cases:
            with self.subTest(deltaS=delta_s, xc=xc, nobeta=nobeta):
                mol = gto.M(
                    atom="C 0 0 0; H 0 0 2.0; H 0 1.7 -0.5",
                    basis="sto-3g",
                    spin=2,
                    unit="Bohr",
                    verbose=0,
                )
                mf = mol.ROKS(xc=xc).average_occ().set(
                    conv_tol=1e-12,
                    conv_tol_grad=1e-9,
                    max_cycle=150,
                    verbose=0,
                )
                mf.grids.level = 0
                mf.kernel()
                self.assertTrue(mf.converged)
                tdobj = NTTDA(mf).set(
                    deltaS=delta_s,
                    nobeta=nobeta,
                    nstates=2,
                    conv_tol=1e-9,
                    lindep=1e-20,
                    max_cycle=200,
                    verbose=0,
                ).run()
                gradient = tdobj.Gradients().set(
                    verbose=0,
                    fixed_grid=True,
                    root_overlap_tol=0.5,
                )
                analytic = gradient.kernel(
                    state=1, atmlst=[1], method="analytic",
                )
                finite_difference = gradient.kernel(
                    state=1,
                    atmlst=[1],
                    method="finite_diff",
                    step=2e-3,
                )
                np.testing.assert_allclose(
                    analytic, finite_difference, atol=5e-5, rtol=0,
                )


if __name__ == "__main__":
    unittest.main()
