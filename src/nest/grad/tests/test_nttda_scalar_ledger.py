#!/usr/bin/env python
"""Scalar-closure tests for both independent NTTDA analytic channels."""

import unittest
from pathlib import Path

import numpy as np

import nest
from pyscf import dft, gto
from nest.nttda import NTTDA



from nest.grad.nttda.delta_s_zero import (  # noqa: E402
    same_spin_action_scalar,
    same_spin_ledger_scalar,
)
from nest.grad.nttda.delta_s_minus_one import (  # noqa: E402
    spin_lowering_action_scalar,
    spin_lowering_ledger_scalar,
)


class SameSpinScalarClosure(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mol = gto.M(
            atom="N 0 0 0; O 0 0 1.20; H 0 0.90 -0.20",
            basis="sto-3g",
            spin=2,
            unit="Bohr",
            verbose=0,
        )

    def test_selected_functionals_eigenvector_and_random_vector(self):
        rng = np.random.default_rng(19)
        for xc in ("HF", "SVWN", "PBE", "TPSS", "M06-2X", "CAM-B3LYP"):
            mf = dft.ROKS(self.mol).set(xc=xc, conv_tol=1e-11, verbose=0)
            mf.grids.level = 0
            mf.kernel()
            self.assertTrue(mf.converged)
            for nobeta in (False, True):
                td = NTTDA(mf).set(
                    deltaS=0,
                    nobeta=nobeta,
                    nstates=3,
                    conv_tol=1e-10,
                    max_cycle=200,
                    verbose=0,
                )
                vind, hdiag = td.gen_vind_sc()
                rows = np.asarray(vind(np.eye(hdiag.size))).reshape(
                    hdiag.size, hdiag.size,
                )
                _energies, eigenvectors = np.linalg.eigh(
                    0.5 * (rows + rows.T),
                )
                vectors = {
                    "root2": eigenvectors[:, 1],
                    "random": rng.normal(size=hdiag.size),
                }
                for vector_kind, vector in vectors.items():
                    with self.subTest(
                            xc=xc, nobeta=nobeta, vector=vector_kind):
                        action = same_spin_action_scalar(td, vector)
                        ledger = same_spin_ledger_scalar(td, vector)
                        self.assertLess(abs(action - ledger), 1e-11)

    def test_nttda_package_does_not_import_satda_gradient_modules(self):
        imported = set(__import__("sys").modules)
        self.assertNotIn("pyscf.grad.tdsatda_delta", imported)
        self.assertNotIn("pyscf.grad.tdsatda_fast", imported)
        self.assertTrue(
            str(Path(nest.grad.nttda.__file__).resolve()).startswith(str(Path(nest.__file__).resolve().parent))
        )

    def test_lowering_channel_uses_an_independent_closed_ledger(self):
        rng = np.random.default_rng(31)
        for xc in ("HF", "SVWN", "PBE", "TPSS", "M06-2X", "CAM-B3LYP"):
            mf = dft.ROKS(self.mol).set(xc=xc, conv_tol=1e-11, verbose=0)
            mf.grids.level = 0
            mf.kernel()
            self.assertTrue(mf.converged)
            for nobeta in (False, True):
                td = NTTDA(mf).set(
                    deltaS=-1,
                    nobeta=nobeta,
                    nstates=3,
                    conv_tol=1e-10,
                    verbose=0,
                ).run()
                _vind, diagonal = td.gen_vind_sfd()
                vectors = (
                    np.asarray(td.xy[1][0]),
                    rng.normal(size=diagonal.size),
                )
                for vector in vectors:
                    with self.subTest(xc=xc, nobeta=nobeta):
                        action = spin_lowering_action_scalar(td, vector)
                        ledger = spin_lowering_ledger_scalar(td, vector)
                        self.assertLess(abs(action - ledger), 1e-11)


if __name__ == "__main__":
    unittest.main()
