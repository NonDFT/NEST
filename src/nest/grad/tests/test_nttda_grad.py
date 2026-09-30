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


if __name__ == "__main__":
    unittest.main()
