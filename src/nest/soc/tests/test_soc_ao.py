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
from pyscf import gto, lib
from nest.soc import soc_ao


def fp(mat):
    return lib.fp(mat)

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
        cls.mol = mol.build()

    @classmethod
    def tearDownClass(cls):
        cls.mol.stdout.close()

    def test_1e_soc_ao(self):
        mf = self.mol.ROKS(xc='HF').run()
        self.assertTrue(mf.converged)

        ref = -0.0035217717150048196 - 0.002071753516350664j
        ao_soc = soc_ao.get_ao_soc(mf, '1e')
        self.assertEqual(ao_soc.shape, (3, self.mol.nao_nr(), self.mol.nao_nr()))
        self.assertAlmostEqual(abs(fp(ao_soc) - ref), 0, delta=1e-11)

    def test_zeff_soc_ao(self):
        mf = self.mol.ROKS(xc='HF').run()
        self.assertTrue(mf.converged)

        ref = -0.002466043752900166 - 0.001450202630169735j
        ao_soc = soc_ao.get_ao_soc(mf, 'Zeff')
        self.assertEqual(ao_soc.shape, (3, self.mol.nao_nr(), self.mol.nao_nr()))
        self.assertAlmostEqual(abs(fp(ao_soc) - ref), 0, delta=1e-11)

    def test_somf_soc_ao(self):
        mf = self.mol.ROKS(xc='HF').run()
        self.assertTrue(mf.converged)

        ref = -0.002331075720748517 - 0.0013735097397886604j
        ao_soc = soc_ao.get_ao_soc(mf, 'SOMF')
        self.assertEqual(ao_soc.shape, (3, self.mol.nao_nr(), self.mol.nao_nr()))
        self.assertAlmostEqual(abs(fp(ao_soc) - ref), 0, delta=1e-9)

    def test_somf_soc_ao_uks(self):
        mf = self.mol.UKS(xc='HF').newton().run()
        self.assertTrue(mf.converged)

        ref = -0.0023423479533036867 - 0.0013759118729275677j
        ao_soc = soc_ao.get_ao_soc(mf, 'SOMF')
        self.assertEqual(ao_soc.shape, (3, self.mol.nao_nr(), self.mol.nao_nr()))
        self.assertAlmostEqual(abs(fp(ao_soc) - ref), 0, delta=1e-9)


if __name__ == '__main__':
    print('Full tests for AO spin-orbit integrals')
    unittest.main()
