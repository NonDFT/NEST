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
from nest import nttda


class KnownValues(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        mol = gto.Mole()
        mol.verbose = 0
        mol.output = '/dev/null'
        mol.atom = """
        O                  0.64372820    0.14077399   -0.04477253
        O                 -0.64862595   -0.12779073   -0.05445498
        H                  1.16027512   -0.65947800    0.36730132
        H                 -1.12109306    0.55561188    0.42651873
        """
        mol.charge = 0
        mol.spin = 2
        mol.basis = '631g'
        cls.mol = mol.build()

    @classmethod
    def tearDownClass(cls):
        cls.mol.stdout.close()

    def test_hf_nttda(self):
        mf = self.mol.ROKS(xc='HF').run()

        ref = np.array([0.26373033968267973, 0.32114587049263738])
        td = mf.NTTDA().set(nstates=2, deltaS=1, nobeta=True).run()
        self.assertTrue(np.all(td.converged))
        self.assertAlmostEqual(abs(td.e - ref).max(), 0, delta=1e-6)

        ref = np.array([-0.25588162251385815, 0.03179164805915535])
        td = mf.NTTDA().set(nstates=2, deltaS=-1, nobeta=False).run()
        self.assertTrue(np.all(td.converged))
        self.assertAlmostEqual(abs(td.e - ref).max(), 0, delta=1e-6)

        ref = np.array([-0.021227306082554027, 0.03681224565830669])
        td = mf.NTTDA().set(nstates=2, deltaS=0, nobeta=False).run()
        self.assertTrue(np.all(td.converged))
        self.assertAlmostEqual(abs(td.e - ref).max(), 0, delta=1e-6)

    def test_svwn_nttda(self):
        mf = self.mol.ROKS(xc='SVWN').run()

        ref = np.array([-0.21136285952298853, 0.022829192982022128])
        td = mf.NTTDA().set(nstates=2, deltaS=-1, nobeta=True).run()
        self.assertTrue(np.all(td.converged))
        self.assertAlmostEqual(abs(td.e - ref).max(), 0, delta=1e-6)

        ref = np.array([-0.0014224229333087768, 0.029907227771976085])
        td = mf.NTTDA().set(nstates=2, deltaS=0, nobeta=True).run()
        self.assertTrue(np.all(td.converged))
        self.assertAlmostEqual(abs(td.e - ref).max(), 0, delta=1e-6)

        ref = np.array([0.2621305574444208, 0.3146577468311684])
        td = mf.NTTDA().set(nstates=2, deltaS=1, nobeta=False).run()
        self.assertTrue(np.all(td.converged))
        self.assertAlmostEqual(abs(td.e - ref).max(), 0, delta=1e-6)

    def test_m062x_nttda(self):
        mf = self.mol.ROKS(xc='M062X').run()

        ref = np.array([-0.24666086824597583, 0.015820053409613927])
        td = mf.NTTDA().set(nstates=2, deltaS=-1, nobeta=True).run()
        self.assertTrue(np.all(td.converged))
        self.assertAlmostEqual(abs(td.e - ref).max(), 0, delta=1e-6)

        ref = np.array([-0.008184446338165025, 0.025150738879015422])
        td = mf.NTTDA().set(nstates=2, deltaS=0, nobeta=False).run()
        self.assertTrue(np.all(td.converged))
        self.assertAlmostEqual(abs(td.e - ref).max(), 0, delta=1e-6)

        ref = np.array([0.26880002289621757, 0.3280851476633962])
        td = mf.NTTDA().set(nstates=2, deltaS=1, nobeta=False).run()
        self.assertTrue(np.all(td.converged))
        self.assertAlmostEqual(abs(td.e - ref).max(), 0, delta=1e-6)

    def test_cam_b3lyp_nttda(self):
        mf = self.mol.ROKS(xc='CAM-B3LYP').run()

        ref = np.array([-0.0044893465927124268, 0.035037117269294718])
        td = mf.NTTDA().set(nstates=2, deltaS=0, nobeta=True).run()
        self.assertTrue(np.all(td.converged))
        self.assertAlmostEqual(abs(td.e - ref).max(), 0, delta=1e-6)

        ref = np.array([0.27155932081326395, 0.32184531828332463])
        td = mf.NTTDA().set(nstates=2, deltaS=1, nobeta=True).run()
        self.assertTrue(np.all(td.converged))
        self.assertAlmostEqual(abs(td.e - ref).max(), 0, delta=1e-6)

        ref = np.array([-0.22362676199942616, 0.02217598445976246])
        td = mf.NTTDA().set(nstates=2, deltaS=-1, nobeta=False).run()
        self.assertTrue(np.all(td.converged))
        self.assertAlmostEqual(abs(td.e - ref).max(), 0, delta=1e-6)


if __name__ == '__main__':
    print('Full tests for noncollinear tensor TDA based on ROKS reference')
    unittest.main()
