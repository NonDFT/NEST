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

from types import SimpleNamespace
import unittest

from pyscf import gto, lib
from nest.soc import soc_ao


try:
    import socutils  # noqa: F401
except ImportError:
    HAS_SOCUTILS = False
else:
    HAS_SOCUTILS = True


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

    def test_somf_amfi_soc_ao(self):
        mf = self.mol.ROKS(xc='HF').run()
        self.assertTrue(mf.converged)

        ref = -0.002258000497354782 - 0.001309642786448594j
        ao_soc = soc_ao.get_ao_soc(mf, 'SOMF_AMFI')
        self.assertEqual(ao_soc.shape, (3, self.mol.nao_nr(), self.mol.nao_nr()))
        self.assertAlmostEqual(abs(fp(ao_soc) - ref), 0, delta=1e-9)

    def test_x2c1e_soc_ao(self):
        ref = -0.0035149790489170775 - 0.002067624769849547j
        ao_soc = soc_ao.get_ao_soc(SimpleNamespace(mol=self.mol), 'X2C1E')
        self.assertEqual(ao_soc.shape, (3, self.mol.nao_nr(), self.mol.nao_nr()))
        self.assertAlmostEqual(abs(fp(ao_soc) - ref), 0, delta=1e-9)

    @unittest.skipUnless(HAS_SOCUTILS, 'socutils is not installed')
    def test_x2camf_soc_ao(self):
        ref = -0.002270958919990373 - 0.00131052918540656j
        ao_soc = soc_ao.get_ao_soc(SimpleNamespace(mol=self.mol), 'X2CAMF')
        self.assertEqual(ao_soc.shape, (3, self.mol.nao_nr(), self.mol.nao_nr()))
        self.assertAlmostEqual(abs(fp(ao_soc) - ref), 0, delta=1e-9)

    @unittest.skipUnless(HAS_SOCUTILS, 'socutils is not installed')
    def test_x2cmp_soc_ao(self):
        ref = -0.0022231653593819713 - 0.0012960492601477023j
        ao_soc = soc_ao.get_ao_soc(SimpleNamespace(mol=self.mol), 'X2CMP')
        self.assertEqual(ao_soc.shape, (3, self.mol.nao_nr(), self.mol.nao_nr()))
        self.assertAlmostEqual(abs(fp(ao_soc) - ref), 0, delta=1e-9)


if __name__ == '__main__':
    print('Full tests for AO spin-orbit integrals')
    unittest.main()
