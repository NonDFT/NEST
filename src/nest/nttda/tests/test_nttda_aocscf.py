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
from nest import aocscf, nttda


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

    def test_svwn_nttda_aocscf(self):
        mf = self.mol.ROKS(xc='SVWN').average_occ()
        mf.conv_tol = 1e-11
        mf.conv_tol_grad = 1e-8
        mf.max_cycle = 200
        mf.grids.level = 3
        mf.grids.prune = None
        mf.small_rho_cutoff = 0.0
        mf.run()

        ref = np.array([-0.21222618958794592, 0.022735913574159522])
        td = mf.NTTDA().set(nstates=2, deltaS=-1, nobeta=True, conv_tol=1e-5, max_cycle=200).run()
        self.assertTrue(np.all(td.converged))
        self.assertAlmostEqual(abs(td.e - ref).max(), 0, delta=1e-6)

        ref = np.array([-0.001800257693000168, 0.030755390462627187])
        td = mf.NTTDA().set(nstates=2, deltaS=0, nobeta=True, conv_tol=1e-5, max_cycle=200).run()
        self.assertTrue(np.all(td.converged))
        self.assertAlmostEqual(abs(td.e - ref).max(), 0, delta=1e-6)

    def test_b3lyp_nttda_aocscf(self):
        mf = self.mol.ROKS(xc='B3LYP').average_occ()
        mf.conv_tol = 1e-11
        mf.conv_tol_grad = 1e-8
        mf.max_cycle = 200
        mf.grids.level = 3
        mf.grids.prune = None
        mf.small_rho_cutoff = 0.0
        mf.run()

        ref = np.array([-0.22131467106409972, 0.020196490053532357])
        td = mf.NTTDA().set(nstates=2, deltaS=-1, nobeta=True, conv_tol=1e-5, max_cycle=200).run()
        self.assertTrue(np.all(td.converged))
        self.assertAlmostEqual(abs(td.e - ref).max(), 0, delta=1e-6)

        ref = np.array([-0.006072490213890671, 0.034052405714217956])
        td = mf.NTTDA().set(nstates=2, deltaS=0, nobeta=True, conv_tol=1e-5, max_cycle=200).run()
        self.assertTrue(np.all(td.converged))
        self.assertAlmostEqual(abs(td.e - ref).max(), 0, delta=1e-6)
