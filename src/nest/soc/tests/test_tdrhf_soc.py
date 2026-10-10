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

import unittest

import numpy as np
from pyscf import gto
from pyscf.data.nist import HARTREE2WAVENUMBER

from nest.soc import TDRHFSOC


def assert_allclose_up_to_sign(testcase, actual, desired, atol):
    try:
        np.testing.assert_allclose(actual, desired, atol=atol, rtol=0)
    except AssertionError as error:
        np.testing.assert_allclose(actual, -desired, atol=atol, rtol=0, err_msg=str(error))


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
        mol.spin = 0
        mol.basis = '631g'
        cls.mol = mol.build()
        cls.mf = cls.mol.RKS(xc='PBE').set(conv_tol=1e-12).run()

    @classmethod
    def tearDownClass(cls):
        cls.mol.stdout.close()

    def test_tda_somf_soc(self):
        singlet = self.mf.TDA().set(
            singlet=True, nstates=3, conv_tol=1e-6, max_cycle=300,
        ).run()
        triplet = self.mf.TDA().set(
            singlet=False, nstates=3, conv_tol=1e-6, max_cycle=300,
        ).run()
        driver = TDRHFSOC(singlet, triplet, soctype='SOMF').run()

        self.assertTrue(self.mf.converged)
        self.assertTrue(np.all(singlet.converged))
        self.assertTrue(np.all(triplet.converged))
        self.assertAlmostEqual(self.mf.e_tot, -151.30837113622292, delta=1e-5)
        np.testing.assert_allclose(singlet.e, [
            0.2134564924537518, 0.2373626411399768, 0.2702813369193999,
        ], atol=1e-5, rtol=0)
        np.testing.assert_allclose(triplet.e, [
            0.1779432733241447, 0.2133608869104915, 0.2294640873382239,
        ], atol=1e-5, rtol=0)

        self.assertEqual(driver.h_soc.shape, (13, 13))
        np.testing.assert_allclose(driver.h_soc, driver.h_soc.conj().T, atol=1e-12)
        np.testing.assert_allclose(
            (driver.e - driver.e.min()).real * HARTREE2WAVENUMBER,
            [
                0.0,
                39053.80720087662,
                39053.831709963764,
                39053.95873442661,
                46819.306973781415,
                46827.398987272376,
                46827.43594320817,
                46855.64498221236,
                50361.930131518464,
                50361.93429081267,
                50362.493268383376,
                52095.240673403314,
                59320.155706283826,
            ],
            atol=1e-4,
            rtol=0,
        )
        assert_allclose_up_to_sign(
            self,
            driver.get_block(4, 0) * HARTREE2WAVENUMBER,
            np.array([
                [-6.530965289131929 + 12.710589139611512j],
                [13.634678990631045j],
                [-6.530965289131929 - 12.710589139611512j],
            ]),
            1e-5,
        )
        assert_allclose_up_to_sign(
            self,
            driver.get_block(4, 1) * HARTREE2WAVENUMBER,
            np.array([
                [1.009014435023905 + 2.9265773721032655j],
                [-4.2036249891559985j],
                [1.009014435023905 - 2.9265773721032655j],
            ]),
            1e-5,
        )

    def test_tddft_somf_soc(self):
        singlet = self.mf.TDDFT().set(
            singlet=True, nstates=3, conv_tol=1e-6, max_cycle=300,
        ).run()
        triplet = self.mf.TDDFT().set(
            singlet=False, nstates=3, conv_tol=1e-6, max_cycle=300,
        ).run()
        driver = TDRHFSOC(singlet, triplet, soctype='SOMF').run()

        self.assertTrue(self.mf.converged)
        self.assertTrue(np.all(singlet.converged))
        self.assertTrue(np.all(triplet.converged))
        self.assertAlmostEqual(self.mf.e_tot, -151.30837113622292, delta=1e-5)
        np.testing.assert_allclose(singlet.e, [
            0.2114594523712823, 0.2363774657780003, 0.2675069553335562,
        ], atol=1e-5, rtol=0)
        np.testing.assert_allclose(triplet.e, [
            0.1764364615611013, 0.2123120145202830, 0.2275201018557039,
        ], atol=1e-5, rtol=0)

        self.assertEqual(driver.h_soc.shape, (13, 13))
        np.testing.assert_allclose(driver.h_soc, driver.h_soc.conj().T, atol=1e-12)
        np.testing.assert_allclose(
            (driver.e - driver.e.min()).real * HARTREE2WAVENUMBER,
            [
                0.0,
                38723.09215099463,
                38723.12012401808,
                38723.24658977237,
                46407.724253763496,
                46597.19604895671,
                46597.24084952394,
                46598.75045006708,
                49935.27727048368,
                49935.28300367386,
                49935.84842633218,
                51879.00205229169,
                58711.25291238867,
            ],
            atol=1e-4,
            rtol=0,
        )
        assert_allclose_up_to_sign(
            self,
            driver.get_block(4, 0) * HARTREE2WAVENUMBER,
            np.array([
                [-6.52087507581652 + 12.466292436586002j],
                [14.511534180185103j],
                [-6.52087507581652 - 12.466292436586002j],
            ]),
            1e-5,
        )
        assert_allclose_up_to_sign(
            self,
            driver.get_block(4, 1) * HARTREE2WAVENUMBER,
            np.array([
                [0.5780840634553324 + 3.471579441709397j],
                [-4.885142953681281j],
                [0.5780840634553324 - 3.471579441709397j],
            ]),
            1e-5,
        )


if __name__ == '__main__':
    print('Full SOC tests for closed-shell TDA and TDDFT')
    unittest.main()
