# Copyright 2021-2024 The PySCF Developers. All Rights Reserved.
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
from unittest import mock
import numpy as np
from pyscf import gto
from nest.nttda import nttda


REFS = {
    'HF': {
        True: {
            -1: np.array([-0.25588162251385949, 0.031791648059151634,
                          0.082164577524099669, 0.10984172789557073,
                          0.14436589862648663]),
            0: np.array([-0.021227306082552837, 0.036812245658307111,
                         0.055828674198231877, 0.10820095281082762,
                         0.13769598373589942]),
            1: np.array([0.26373033968267973, 0.32114587049263738,
                         0.35767192060413755, 0.4089546816647468,
                         0.48418822465436356]),
        },
        False: {
            -1: np.array([-0.25588162251385815, 0.03179164805915535,
                          0.08216457752408901, 0.10984172789556618,
                          0.14436589862650168]),
            0: np.array([-0.021227306082554027, 0.03681224565830669,
                         0.05582867419822887, 0.10820095281083697,
                         0.13769598373589134]),
            1: np.array([0.26373033968267307, 0.32114587049263676,
                         0.35767192060412084, 0.4089546816647443,
                         0.48418822465431444]),
        },
    },
    'SVWN': {
        True: {
            -1: np.array([-0.21136285952298853, 0.022829192982022128,
                          0.04449709298041335, 0.070334528481137998,
                          0.11581794978093166]),
            0: np.array([-0.0014224229333087768, 0.029907227771976085,
                         0.042159504595931208, 0.087581948278004515,
                         0.17827490921649417]),
            1: np.array([0.26097145556097057, 0.31399118616119204,
                         0.40046031718535502, 0.44773897177486011,
                         0.45191690809443946]),
        },
        False: {
            -1: np.array([-0.21170979048359168, 0.023046236405179832,
                          0.04399445674703403, 0.07151698987123586,
                          0.1149177949908444]),
            0: np.array([-0.0014102920144926014, 0.029534076286594643,
                         0.043327478728623726, 0.08675602648118973,
                         0.17868955257035488]),
            1: np.array([0.2621305574444208, 0.3146577468311684,
                         0.400854855533031, 0.4488089982219909,
                         0.45217145231139155]),
        },
    },
    'M062X': {
        True: {
            -1: np.array([-0.24666086824597583, 0.015820053409613927,
                          0.050190722681826144, 0.071795073579681096,
                          0.12358842176137239]),
            0: np.array([-0.0066422638316957381, 0.028055776231764321,
                         0.034831792351000868, 0.097283193576694127,
                         0.16108162207164683]),
            1: np.array([0.277635913239132, 0.33395796939250971,
                         0.38888717645852439, 0.44267039406168623,
                         0.49110742356819381]),
        },
        False: {
            -1: np.array([-0.24280053851498867, 0.011530030283297799,
                          0.05005354330269396, 0.06762698114448712,
                          0.12639763640154533]),
            0: np.array([-0.008184446338165025, 0.025150738879015422,
                         0.032777031664227074, 0.09911876211938828,
                         0.15953063092372488]),
            1: np.array([0.26880002289621757, 0.3280851476633962,
                         0.3822461897656717, 0.4389979233432141,
                         0.4818601324603382]),
        },
    },
    'CAM-B3LYP': {
        True: {
            -1: np.array([-0.22468903600466386, 0.022443972864282041,
                          0.053034041517141139, 0.079464935567422901,
                          0.12134548968286102]),
            0: np.array([-0.0044893465927124268, 0.035037117269294718,
                         0.043274626762097285, 0.096035968020390092,
                         0.17133618259284775]),
            1: np.array([0.27155932081326395, 0.32184531828332463,
                         0.38819254300788419, 0.43485814799250122,
                         0.47810311079140677]),
        },
        False: {
            -1: np.array([-0.22362676199942616, 0.02217598445976246,
                          0.0521859034687622, 0.08054218557201234,
                          0.12099781181846828]),
            0: np.array([-0.004488754315208394, 0.03433453230586325,
                         0.044209793487172355, 0.09543419827896533,
                         0.17172220940564042]),
            1: np.array([0.27064837890318744, 0.32121564431676974,
                         0.3866085753082913, 0.43353862650338093,
                         0.47757913242638816]),
        },
    },
}


class KnownValues(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        mol = gto.Mole()
        mol.verbose = 0
        mol.output = '/dev/null'
        mol.atom = '''
        O                  0.64372820    0.14077399   -0.04477253
        O                 -0.64862595   -0.12779073   -0.05445498
        H                  1.16027512   -0.65947800    0.36730132
        H                 -1.12109306    0.55561188    0.42651873
        '''
        mol.charge = 0
        mol.spin = 2
        mol.basis = '631g'
        mol.symmetry = True
        cls.mol = mol.build()

    @classmethod
    def tearDownClass(cls):
        cls.mol.stdout.close()

    def _check_functional(self, xc):
        mf = self.mol.ROKS(xc=xc).run()
        for nobeta, refs_by_delta_s in REFS[xc].items():
            for delta_s, ref in refs_by_delta_s.items():
                with self.subTest(xc=xc, nobeta=nobeta, deltaS=delta_s):
                    td = nttda.NTTDA(mf)
                    td.nstates = 5
                    td.deltaS = delta_s
                    td.nobeta = nobeta
                    td.verbose = 0
                    td.kernel()
                    self.assertTrue(np.all(td.converged))
                    np.testing.assert_allclose(td.e, ref, atol=1e-6, rtol=0)

    def test_hf_nttda(self):
        self._check_functional('HF')

    def test_svwn_nttda(self):
        self._check_functional('SVWN')

    def test_m062x_nttda(self):
        self._check_functional('M062X')

    def test_cam_b3lyp_nttda(self):
        self._check_functional('CAM-B3LYP')

    def test_mo_grid_fxc1_vind_matches_ao(self):
        rng = np.random.default_rng(12)
        for xc in ('BLYP', 'TPSS'):
            mf = self.mol.ROKS(xc=xc).run()
            for delta_s in (0, -1):
                with self.subTest(xc=xc, deltaS=delta_s):
                    td0 = nttda.NTTDA(mf)
                    td0.deltaS = delta_s
                    td0.verbose = 0
                    td1 = nttda.NTTDA(mf)
                    td1.deltaS = delta_s
                    td1.verbose = 0
                    with mock.patch.object(nttda, 'MO_GRID_FXC1', False):
                        vind0, hdiag0 = (td0.gen_vind_sc() if delta_s == 0
                                         else td0.gen_vind_sfd())
                    with mock.patch.object(nttda, 'MO_GRID_FXC1', True):
                        vind1, hdiag1 = (td1.gen_vind_sc() if delta_s == 0
                                         else td1.gen_vind_sfd())
                    np.testing.assert_allclose(hdiag1, hdiag0, atol=1e-10, rtol=0)
                    zs = rng.standard_normal((2, hdiag0.size))
                    np.testing.assert_allclose(vind1(zs), vind0(zs), atol=1e-9, rtol=0)


if __name__ == "__main__":
    print("Full Tests for noncollinear tensor TDA based on ROKS reference")
    unittest.main()
