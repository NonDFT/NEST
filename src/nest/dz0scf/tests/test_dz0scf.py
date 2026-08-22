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

import unittest
import numpy as np

from pyscf import gto
from nest.dz0scf import DZ0SCF
from nest.nttda import NTTDA


class KnownValues(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mol = gto.M(
            atom="""
O   0.64372820   0.14077399  -0.04477253
O  -0.64862595  -0.12779073  -0.05445498
H   1.16027512  -0.65947800   0.36730132
H  -1.12109306   0.55561188   0.42651873
""",
            basis="6-31g",
            unit="Angstrom",
            charge=0,
            spin=2,
            symmetry=False,
            verbose=0,
        )

    def test_svwn_dz0scf(self):
        mf = DZ0SCF(self.mol, xc="SVWN")
        mf.conv_tol = 1e-11
        mf.conv_tol_grad = 1e-8
        mf.max_cycle = 200
        mf.grids.level = 3
        mf.grids.prune = None
        mf.small_rho_cutoff = 0.0
        mf.kernel()

        self.assertTrue(mf.converged)

        e_dz0_ref = -150.15324131943828
        e_high_spin_ref = -150.18135492533739

        self.assertAlmostEqual(
            mf.e_tot,
            e_dz0_ref,
            delta=1e-7,
        )
        self.assertAlmostEqual(
            mf.high_spin_energy(),
            e_high_spin_ref,
            delta=1e-7,
        )

        td_s = NTTDA(mf)
        td_s.deltaS = -1
        td_s.nstates = 2
        td_s.nobeta = True
        td_s.conv_tol = 1e-5
        td_s.max_cycle = 200

        omega_s, _ = td_s.kernel()

        omega_s_ref = np.array([
            -0.21222618958794592,
             0.022735913574159522,
        ])

        self.assertTrue(np.all(np.asarray(td_s.converged)))
        np.testing.assert_allclose(
            np.asarray(omega_s),
            omega_s_ref,
            rtol=0.0,
            atol=1e-6,
        )

        td_t = NTTDA(mf)
        td_t.deltaS = 0
        td_t.nstates = 2
        td_t.nobeta = True
        td_t.conv_tol = 1e-5
        td_t.max_cycle = 200

        omega_t, _ = td_t.kernel()

        omega_t_ref = np.array([
            -0.001800257693000168,
             0.030755390462627187,
        ])

        self.assertTrue(np.all(np.asarray(td_t.converged)))
        np.testing.assert_allclose(
            np.asarray(omega_t),
            omega_t_ref,
            rtol=0.0,
            atol=1e-6,
        )

    def test_b3lyp_dz0scf(self):
        mf = DZ0SCF(self.mol, xc="B3LYP")
        mf.conv_tol = 1e-11
        mf.conv_tol_grad = 1e-8
        mf.max_cycle = 200
        mf.grids.level = 3
        mf.grids.prune = None
        mf.small_rho_cutoff = 0.0
        mf.kernel()

        self.assertTrue(mf.converged)

        e_dz0_ref = -151.18245418239550
        e_high_spin_ref = -151.25619865161033

        self.assertAlmostEqual(
            mf.e_tot,
            e_dz0_ref,
            delta=1e-7,
        )
        self.assertAlmostEqual(
            mf.high_spin_energy(),
            e_high_spin_ref,
            delta=1e-7,
        )

        td_s = NTTDA(mf)
        td_s.deltaS = -1
        td_s.nstates = 2
        td_s.nobeta = True
        td_s.conv_tol = 1e-5
        td_s.max_cycle = 200

        omega_s, _ = td_s.kernel()

        omega_s_ref = np.array([
            -0.22131467106409972,
             0.020196490053532357,
        ])

        self.assertTrue(np.all(np.asarray(td_s.converged)))
        np.testing.assert_allclose(
            np.asarray(omega_s),
            omega_s_ref,
            rtol=0.0,
            atol=1e-6,
        )

        td_t = NTTDA(mf)
        td_t.deltaS = 0
        td_t.nstates = 2
        td_t.nobeta = True
        td_t.conv_tol = 1e-5
        td_t.max_cycle = 200

        omega_t, _ = td_t.kernel()

        omega_t_ref = np.array([
            -0.006072490213890671,
             0.034052405714217956,
        ])

        self.assertTrue(np.all(np.asarray(td_t.converged)))
        np.testing.assert_allclose(
            np.asarray(omega_t),
            omega_t_ref,
            rtol=0.0,
            atol=1e-6,
        )    