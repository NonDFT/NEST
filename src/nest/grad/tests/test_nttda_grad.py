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
        # Use the complete 14-dimensional raising space in this tiny molecule.
        # This keeps reference and displaced roots below the 1e-9 residual
        # tolerance without marginal Davidson residuals.
        nstates = 3
        if delta_s == 1:
            nstates = np.count_nonzero(mf.mo_occ == 2) * np.count_nonzero(mf.mo_occ == 0)
        tdobj = NTTDA(mf).set(
            deltaS=delta_s,
            nobeta=nobeta,
            nstates=nstates,
            conv_tol=1e-9,
            lindep=1e-20,
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

    def test_delta_s_plus_one_hf_lda_gga_mgga_hybrid_and_rsh(self):
        cases = (
            ("HF", False),
            ("SVWN", False),
            ("PBE", False),
            ("TPSS", False),
            ("M06-2X", False),
            ("M06-2X", True),
            ("CAM-B3LYP", False),
            ("CAM-B3LYP", True),
        )
        for xc, nobeta in cases:
            with self.subTest(xc=xc, nobeta=nobeta):
                self.compare_public_gradient(
                    xc, delta_s=1, nobeta=nobeta, threshold=1e-5,
                )

    def test_gradient_inherits_td_memory_limit(self):
        mf = dft.ROKS(self.molecule()).set(max_memory=128)
        tdobj = NTTDA(mf).set(max_memory=64)
        self.assertEqual(tdobj.Gradients().max_memory, 64)

    def test_gradient_parts_and_scanner_keep_total_energy_convention(self):
        for delta_s in (0, 1):
            with self.subTest(delta_s=delta_s):
                tdobj = self.make_td("HF", delta_s=delta_s)
                gradient = tdobj.Gradients()
                reference = gradient.kernel(state=0)
                total = gradient.kernel(state=2)
                np.testing.assert_allclose(gradient.grad_nuc(), reference, atol=1e-10, rtol=0)
                np.testing.assert_allclose(
                    gradient.grad_elec(tdobj.xy[1]) + reference, total, atol=1e-10, rtol=0,
                )
                scanner = gradient.as_scanner(state=2)
                energy, scanned = scanner(tdobj.mol.copy())
                self.assertTrue(scanner.converged)
                self.assertAlmostEqual(energy, tdobj.e_tot[1], places=8)
                np.testing.assert_allclose(scanned, total, atol=1e-8, rtol=0)
                energy, scanned = scanner(tdobj.mol.copy(), state=0)
                self.assertTrue(scanner.converged)
                self.assertAlmostEqual(energy, tdobj._scf.e_tot, places=8)
                np.testing.assert_allclose(scanned, reference, atol=1e-8, rtol=0)

    def test_gradient_without_independent_orbital_rotations(self):
        mol = gto.M(
            atom="H 0 0 0; H 0 0 1.4", basis="sto-3g",
            spin=2, unit="Bohr", verbose=0,
        )
        mf = mol.ROKS(xc="HF").run(conv_tol=1e-12)
        # Both orbitals are open: there are no CO, CV or OV rotations.
        np.testing.assert_array_equal(mf.mo_occ, [1, 1])
        tdobj = NTTDA(mf).set(
            deltaS=-1, nstates=2, conv_tol=1e-9, verbose=0,
        ).run()
        gradient = tdobj.Gradients()
        analytic = gradient.kernel(state=1)
        self.assertEqual(gradient.nttda_details.zvector.size, 0)
        self.assertEqual(gradient.nttda_details.residual, 0)
        finite_difference = gradient.kernel(
            state=1, method="finite_diff", step=2e-4,
        )
        np.testing.assert_allclose(analytic, finite_difference, atol=1e-7, rtol=0)


if __name__ == "__main__":
    unittest.main()
