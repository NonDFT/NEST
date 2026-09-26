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

"""Public NTTDA analytic-gradient acceptance tests."""

import unittest
from pathlib import Path

import numpy as np




from pyscf import dft, gto


from nest.nttda import NTTDA  # noqa: E402


class NTTDAGradientAcceptance(unittest.TestCase):
    @staticmethod
    def molecule():
        return gto.M(
            atom="N 0 0 0; O 0 0 1.20; H 0 0.90 -0.20",
            basis="sto-3g",
            spin=2,
            unit="Bohr",
            verbose=0,
        )

    def make_td(self, xc, delta_s, nobeta=False):
        mf = dft.ROKS(self.molecule()).set(
            xc=xc,
            conv_tol=1e-14,
            conv_tol_grad=1e-11,
            max_cycle=200,
            verbose=0,
        )
        mf.grids.level = 0
        mf.kernel()
        self.assertTrue(mf.converged)
        tdobj = NTTDA(mf).set(
            deltaS=delta_s,
            nobeta=nobeta,
            nstates=3,
            conv_tol=1e-9,
            max_cycle=200,
            verbose=0,
        ).run()
        self.assertGreaterEqual(len(tdobj.xy), 2)
        return tdobj

    def compare_public_gradient(self, xc, delta_s, nobeta, threshold):
        tdobj = self.make_td(xc, delta_s, nobeta=nobeta)
        gradient = tdobj.Gradients().set(
            verbose=0,
            fixed_grid=True,
            root_overlap_tol=0.5,
        )
        analytic = gradient.kernel(state=2, method="analytic")
        self.assertLess(gradient.nttda_details.residual, 1e-8)
        finite_difference = gradient.kernel(
            state=2, method="finite_diff", step=2e-4,
        )
        error = np.max(np.abs(analytic - finite_difference))
        self.assertLess(error, threshold)

    def test_delta_s_zero_hf_lda_gga_mgga_hybrid_and_rsh(self):
        cases = (
            ("HF", False, 3e-5),
            ("SVWN", False, 1e-5),
            ("PBE", False, 1e-5),
            ("TPSS", False, 1e-5),
            ("M06-2X", False, 1e-5),
            ("M06-2X", True, 1e-5),
            ("CAM-B3LYP", False, 1e-5),
        )
        for xc, nobeta, threshold in cases:
            with self.subTest(xc=xc, nobeta=nobeta):
                self.compare_public_gradient(
                    xc, delta_s=0, nobeta=nobeta, threshold=threshold,
                )

    def test_delta_s_minus_one_shares_the_independent_driver(self):
        for xc, nobeta in (("PBE", False), ("M06-2X", True)):
            with self.subTest(xc=xc, nobeta=nobeta):
                self.compare_public_gradient(
                    xc, delta_s=-1, nobeta=nobeta, threshold=1e-5,
                )

    def test_delta_s_plus_one_rejects_analytic_and_keeps_finite_difference(self):
        tdobj = self.make_td("HF", delta_s=1)
        gradient = tdobj.Gradients().set(verbose=0)
        with self.assertRaisesRegex(NotImplementedError, "deltaS=1"):
            gradient.kernel(state=1, method="analytic")
        finite_difference = gradient.kernel(
            state=1,
            atmlst=[0],
            method="finite_diff",
            step=1e-3,
        )
        self.assertTrue(np.all(np.isfinite(finite_difference)))

    def test_scanner_returns_the_selected_energy_and_gradient(self):
        for average_occ in (False, True):
            for delta_s in (-1, 0):
                with self.subTest(average_occ=average_occ, deltaS=delta_s):
                    mol = gto.M(
                        atom="C 0 0 0; H 0 0 2.0; H 0 1.7 -0.5",
                        basis="sto-3g", spin=2, unit="Bohr", verbose=0,
                    )
                    mf = mol.ROKS(xc="HF")
                    if average_occ:
                        mf = mf.average_occ()
                    mf.set(conv_tol=1e-13, conv_tol_grad=1e-10).run()
                    td = NTTDA(mf).set(
                        deltaS=delta_s, nstates=2, conv_tol=1e-9, lindep=1e-18,
                    ).run()
                    grad = td.Gradients().set(state=2)
                    expected = grad.kernel()
                    scanner = grad.as_scanner()
                    energy, derivative = scanner(mol)
                    self.assertTrue(scanner.converged)
                    self.assertEqual(np.ndim(energy), 0)
                    self.assertAlmostEqual(energy, td.e_tot[1], delta=1e-9)
                    np.testing.assert_allclose(derivative, expected, atol=1e-7, rtol=0)

                    coords = mol.atom_coords()
                    coords[1, 2] += 0.02
                    moved = mol.set_geom_(coords, unit="Bohr", inplace=False)
                    energy, derivative = scanner(moved)
                    self.assertTrue(scanner.converged)
                    self.assertAlmostEqual(energy, scanner.base.e_tot[1], delta=1e-12)
                    np.testing.assert_allclose(
                        derivative, scanner.base.Gradients().kernel(state=2), atol=1e-9, rtol=0,
                    )

                    energy, derivative = scanner(moved, state=0)
                    self.assertTrue(scanner.converged)
                    self.assertAlmostEqual(energy, scanner.base._scf.e_tot, delta=1e-12)
                    np.testing.assert_allclose(
                        derivative, scanner.base._scf.nuc_grad_method().kernel(), atol=1e-9, rtol=0,
                    )

    def test_symmetry_and_atom_selection(self):
        mol = gto.M(
            atom="H 0 .934473 -.588078; H 0 -.934473 -.588078; C 0 0 0; O 0 0 1.221104",
            basis="sto-3g", spin=2, symmetry=True, verbose=0,
        )
        atoms = [3, 0]
        for average_occ in (False, True):
            mf = mol.ROKS(xc="HF")
            if average_occ:
                mf = mf.average_occ()
            mf.set(conv_tol=1e-12, conv_tol_grad=1e-9, max_cycle=150).run()
            for delta_s in (-1, 0):
                with self.subTest(average_occ=average_occ, deltaS=delta_s):
                    td = mf.NTTDA().set(
                        deltaS=delta_s, nstates=2, conv_tol=1e-9, lindep=1e-18,
                    ).run()
                    grad = td.Gradients()
                    full = grad.kernel(state=1)
                    self.assertEqual(full.shape, (4, 3))
                    np.testing.assert_allclose(full.sum(axis=0), 0, atol=1e-7, rtol=0)
                    np.testing.assert_allclose(
                        grad.kernel(atmlst=atoms), full[atoms], atol=1e-10, rtol=0,
                    )
                    np.testing.assert_allclose(
                        grad.kernel(state=0, atmlst=atoms),
                        mf.nuc_grad_method().kernel()[atoms], atol=1e-10, rtol=0,
                    )


if __name__ == "__main__":
    unittest.main()
