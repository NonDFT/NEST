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
from nest import aocscf


class KnownValues(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        mol = gto.Mole()
        mol.verbose = 0
        mol.output = '/dev/null'
        mol.atom = '''
        O   0.64372820   0.14077399  -0.04477253
        O  -0.64862595  -0.12779073  -0.05445498
        H   1.16027512  -0.65947800   0.36730132
        H  -1.12109306   0.55561188   0.42651873
        '''
        mol.spin = 2
        mol.basis = '631g'
        cls.mol = mol.build()

    @classmethod
    def tearDownClass(cls):
        cls.mol.stdout.close()

    def test_svwn_aocscf(self):
        mf = self.mol.ROKS(xc='SVWN').average_occ()
        mf.conv_tol = 1e-11
        mf.conv_tol_grad = 1e-8
        mf.max_cycle = 200
        mf.grids.level = 3
        mf.grids.prune = None
        mf.small_rho_cutoff = 0.0
        mf.run()

        self.assertTrue(mf.converged)
        self.assertAlmostEqual(mf.e_avg_occ, -150.15324131943828, delta=1e-7)
        self.assertAlmostEqual(mf.e_tot, -150.18135492533739, delta=1e-7)

    def test_b3lyp_aocscf(self):
        mf = self.mol.ROKS(xc='B3LYP').average_occ()
        mf.conv_tol = 1e-11
        mf.conv_tol_grad = 1e-8
        mf.max_cycle = 200
        mf.grids.level = 3
        mf.grids.prune = None
        mf.small_rho_cutoff = 0.0
        mf.run()

        self.assertTrue(mf.converged)
        self.assertAlmostEqual(mf.e_avg_occ, -151.18245418239550, delta=1e-7)
        self.assertAlmostEqual(mf.e_tot, -151.25619865161033, delta=1e-7)

    def test_svwn_aocscf_newton(self):
        mf = self.mol.ROKS(xc='SVWN').average_occ()
        mf.conv_tol = 1e-11
        mf.conv_tol_grad = 1e-6
        mf.max_cycle = 200
        mf.grids.level = 3
        mf.grids.prune = None
        mf.small_rho_cutoff = 0.0
        mf = mf.newton().run()

        self.assertTrue(mf.converged)
        self.assertAlmostEqual(mf.e_avg_occ, -150.15324131943828, delta=1e-7)
        self.assertAlmostEqual(mf.e_tot, -150.18135492533739, delta=1e-6)

    def test_b3lyp_aocscf_newton_symmetry(self):
        mol = gto.M(
            atom='O 0 0 0; H 0 -.757 .587; H 0 .757 .587',
            basis='6-31g', spin=2, symmetry=True, verbose=0)
        mf = mol.ROKS(xc='B3LYP').average_occ()
        mf.conv_tol = 1e-11
        mf.conv_tol_grad = 1e-6
        mf.grids.level = 0
        mf = mf.newton().run()

        self.assertTrue(mf.converged)
        self.assertAlmostEqual(mf.e_avg_occ, -76.02438483647123, delta=1e-7)
        self.assertAlmostEqual(mf.e_tot, -76.11623061296555, delta=1e-6)

    def test_sfx2c1e_aocscf(self):
        reference = self.mol.ROKS(xc='B3LYP').sfx2c1e()
        reference.grids.level = 0
        mf = reference.average_occ().run()

        self.assertTrue(mf.converged)
        self.assertIs(mf.with_x2c, reference.with_x2c)
        self.assertAlmostEqual(mf.e_tot, reference.energy_tot(dm=mf.make_rdm1()), delta=1e-9)
        self.assertAlmostEqual(mf.e_avg_occ, mf.energy_tot(), delta=1e-9)

    def test_aocscf_df_sfx2c1e(self):
        reference = self.mol.ROKS(xc='B3LYP').density_fit().sfx2c1e()
        reference.grids.level = 0
        mf = self.mol.ROKS(xc='B3LYP').average_occ().density_fit().sfx2c1e()
        mf.grids = reference.grids
        mf.run()

        self.assertTrue(mf.converged)
        self.assertAlmostEqual(mf.e_tot, reference.energy_tot(dm=mf.make_rdm1()), delta=1e-9)
        self.assertAlmostEqual(mf.e_avg_occ, mf.energy_tot(), delta=1e-9)

    def test_newton_aocscf_sfx2c1e(self):
        reference = self.mol.ROKS(xc='B3LYP').sfx2c1e()
        reference.grids.level = 0
        mf = self.mol.ROKS(xc='B3LYP').newton().average_occ().sfx2c1e()
        mf.grids = mf._scf.grids = reference.grids
        mf.run()

        self.assertTrue(mf.converged)
        self.assertIs(mf.with_x2c, mf._scf.with_x2c)
        self.assertAlmostEqual(mf.e_tot, reference.energy_tot(dm=mf.make_rdm1()), delta=1e-9)
        self.assertAlmostEqual(mf.e_avg_occ, mf.remove_soscf().energy_tot(), delta=1e-9)
