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

"""Regression tests for the Dz0SCF high-spin-reference analytic gradient.

The reference values were generated with PySCF 2.13.0.  The analytic-gradient
implementation was independently checked against central finite differences.
NH2/PBE/6-31G exercises a 44-dimensional response space, while triplet
CH2/B3LYP/STO-3G explicitly exercises a singly occupied O space of dimension
two.
"""

import unittest

import numpy as np
from pyscf import gto

from nest.dz0scf import DZ0SCF
from nest.grad.dz0scf import Gradients


GRADIENT_ATOL = 1.0e-6
TRANSLATION_ATOL = 1.0e-6
STATIONARITY_TOL = 1.0e-7
Z_RESIDUAL_TOL = 1.0e-8


NH2_PBE_GRAD = np.array(
    [
        [-2.427124373439239e-15, -3.595373998342540e-02, 6.282335070410562e-02],
        [1.181596094619805e-15, -1.750053024027649e-02, -2.075848019567249e-02],
        [1.223560123578276e-15, 5.345440088313257e-02, -4.206455607347775e-02],
    ]
)


CH2_B3LYP_OPEN2_GRAD = np.array(
    [
        [-7.664563046880733e-04, -2.770013672468408e-02, 5.335666299824815e-02],
        [2.403193394966376e-03, -6.315495031449209e-02, -2.395287280504814e-02],
        [-1.636476595240756e-03, 9.085495314029288e-02, -2.940373960552241e-02],
    ]
)


def run_dz0_gradient(atom, spin, xc, basis):
    """Run one tightly converged and reproducible Dz0SCF gradient."""
    mol = gto.M(
        atom=atom,
        unit="Angstrom",
        basis=basis,
        charge=0,
        spin=spin,
        symmetry=False,
        verbose=0,
        output="/dev/null",
    )

    mf = DZ0SCF(mol, xc=xc)
    mf.conv_tol = 1.0e-12
    mf.conv_tol_grad = 1.0e-9
    mf.max_cycle = 120
    mf.grids.level = 5
    mf.grids.prune = None
    mf.small_rho_cutoff = 0.0
    mf.kernel()
    if not mf.converged:
        mol.stdout.close()
        raise RuntimeError(f"{xc}/{basis} Dz0SCF did not converge")

    # Exercise the public API installed on the Dz0SCF class/mixin.
    grad_obj = mf.nuc_grad_method()
    if not isinstance(grad_obj, Gradients):
        mol.stdout.close()
        raise TypeError(
            "DZ0SCF.nuc_grad_method() did not return "
            "nest.grad.dz0scf.Gradients"
        )

    grad_obj.conv_tol = 1.0e-10
    grad_obj.max_cycle = 120
    grad_obj.restart = 50
    gradient = np.asarray(grad_obj.kernel())
    return mol, mf, grad_obj, gradient


class KnownValues(unittest.TestCase):
    def assert_dz0_result(
        self,
        mf,
        grad_obj,
        gradient,
        reference,
        expected_space,
    ):
        """Check the orbital space, response equations, and nuclear gradient."""
        occupations = np.asarray(mf.mo_occ)
        n_closed = int(np.count_nonzero(np.isclose(occupations, 2.0)))
        n_open = int(np.count_nonzero(np.isclose(occupations, 1.0)))
        n_virtual = int(np.count_nonzero(np.isclose(occupations, 0.0)))
        observed_space = (n_closed, n_open, n_virtual, grad_obj._space.size)

        self.assertEqual(observed_space, expected_space)
        self.assertLess(
            float(np.max(np.abs(grad_obj.g_dz0))),
            STATIONARITY_TOL,
        )

        z_residual = (
            grad_obj.hessian_vector_product(grad_obj.z) - grad_obj.g_hs
        )
        self.assertLess(float(np.linalg.norm(z_residual)), Z_RESIDUAL_TOL)

        np.testing.assert_allclose(
            np.sum(gradient, axis=0),
            np.zeros(3),
            rtol=0.0,
            atol=TRANSLATION_ATOL,
        )
        np.testing.assert_allclose(
            gradient,
            reference,
            rtol=0.0,
            atol=GRADIENT_ATOL,
        )

    def test_pbe_nh2_gradient(self):
        """Check all NH2/PBE/6-31G components in a 44-D response space."""
        mol, mf, grad_obj, gradient = run_dz0_gradient(
            atom="""
                N   0.000000  -0.040000   0.000000
                H   0.000000   0.780000   0.590000
                H   0.000000  -0.860000   0.520000
            """,
            spin=1,
            xc="PBE",
            basis="6-31g",
        )
        try:
            self.assert_dz0_result(
                mf,
                grad_obj,
                gradient,
                NH2_PBE_GRAD,
                expected_space=(4, 1, 8, 44),
            )
        finally:
            mol.stdout.close()

    def test_b3lyp_ch2_two_open_orbitals_gradient(self):
        """Check all CH2 components with a two-dimensional open-shell space."""
        mol, mf, grad_obj, gradient = run_dz0_gradient(
            atom="""
                C    0.020000  -0.030000   0.010000
                H   -0.020000   0.800000   0.620000
                H    0.030000  -0.910000   0.500000
            """,
            spin=2,
            xc="B3LYP",
            basis="sto-3g",
        )
        try:
            self.assert_dz0_result(
                mf,
                grad_obj,
                gradient,
                CH2_B3LYP_OPEN2_GRAD,
                expected_space=(3, 2, 2, 16),
            )
        finally:
            mol.stdout.close()


if __name__ == "__main__":
    unittest.main()
