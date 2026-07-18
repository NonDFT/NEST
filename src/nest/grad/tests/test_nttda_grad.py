#!/usr/bin/env python
"""NEST public NTTDA analytic-gradient acceptance tests."""

import unittest

import numpy as np

from pyscf import dft, gto

from nest.nttda.nttda import NTTDA


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
                    xc, delta_s=1, nobeta=nobeta, threshold=threshold,
                )

    def test_delta_s_plus_one_closed_shell_to_triplet(self):
        mol = gto.M(
            atom="Li 0 0 0; H 0 0 1.60",
            basis="sto-3g",
            spin=0,
            unit="Bohr",
            verbose=0,
        )
        for xc in ("HF", "PBE"):
            with self.subTest(xc=xc):
                mf = dft.ROKS(mol).set(
                    xc=xc,
                    conv_tol=1e-13,
                    conv_tol_grad=1e-10,
                    max_cycle=200,
                    verbose=0,
                )
                mf.grids.level = 0
                mf.kernel()
                self.assertTrue(mf.converged)
                tdobj = NTTDA(mf).set(
                    deltaS=1,
                    nstates=2,
                    conv_tol=1e-10,
                    max_cycle=200,
                    verbose=0,
                ).run()
                gradient = tdobj.Gradients().set(
                    verbose=0,
                    fixed_grid=True,
                    cphf_conv_tol=1e-10,
                )
                analytic = gradient.kernel(
                    state=1, atmlst=[0], method="analytic",
                )
                finite_difference = gradient.kernel(
                    state=1,
                    atmlst=[0],
                    method="finite_diff",
                    step=2e-4,
                )
                np.testing.assert_allclose(
                    analytic, finite_difference, atol=1e-6, rtol=0,
                )


if __name__ == "__main__":
    unittest.main()
