# Copyright 2026 The NEST Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

import unittest
import numpy as np
from pyscf import gto
from pyscf.data.nist import HARTREE2WAVENUMBER
from nest import sftda


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
        mol.spin = 2
        mol.basis = '631g'
        cls.mol = mol.build()

    @classmethod
    def tearDownClass(cls):
        cls.mol.stdout.close()

    def test_roks_sftddft_soc(self):
        mf = self.mol.ROKS(xc='SVWN').run()
        td = sftda.TDDFT_SF(mf).set(
            extype=1, collinear='mcol', collinear_samples=50, nstates=3,
        ).run()
        driver = td.SOC(soctype='SOMF')
        driver.kernel()

        self.assertTrue(mf.converged)
        self.assertTrue(np.all(td.converged))
        self.assertAlmostEqual(mf.e_tot, -150.18173594947856, delta=1e-9)
        np.testing.assert_allclose(td.e, [
            -0.2107471654542317, -0.0015844442063221, 0.0245430214421581,
        ], atol=1e-8, rtol=0)
        np.testing.assert_allclose(td.spin_square(), [
            0.0011758949708333688, 2.0012261115407766, 0.034155774522082627,
        ], atol=1e-8, rtol=0)
        self.assertEqual(driver.h_soc.shape, (5, 5))
        np.testing.assert_allclose(driver.h_soc, driver.h_soc.conj().T, atol=1e-12)
        np.testing.assert_allclose((driver.e - driver.e.min()).real * HARTREE2WAVENUMBER, [
            0.0, 45905.916322187106, 45905.91746253988, 45905.92316008696, 51640.23516406501,
        ], atol=1e-5, rtol=0)
        assert_allclose_up_to_sign(self, driver.get_block(1, 0) * HARTREE2WAVENUMBER, np.array([
            [0.4626543873150336 - 6.686744761741945j],
            [0.0 - 14.2382866761041j],
            [0.4626543873150336 + 6.686744761741945j],
        ]), 1e-8)

    def test_uks_sftddft_soc(self):
        mf = self.mol.UKS(xc='SVWN').run()
        td = sftda.TDDFT_SF(mf).set(
            extype=1, collinear='mcol', collinear_samples=50, nstates=3,
        ).run()
        driver = td.SOC(soctype='SOMF')
        driver.kernel()

        self.assertTrue(mf.converged)
        self.assertTrue(np.all(td.converged))
        self.assertAlmostEqual(mf.e_tot, -150.18252681880003, delta=1e-9)
        np.testing.assert_allclose(td.e, [
            -0.20907621837286508, 0.0000011484871538744751, 0.026028246454667784,
        ], atol=1e-8, rtol=0)
        np.testing.assert_allclose(td.spin_square(), [
            0.0027459512964301, 2.0084594505481292, 0.0400313245672246,
        ], atol=1e-8, rtol=0)
        self.assertEqual(driver.h_soc.shape, (5, 5))
        np.testing.assert_allclose(driver.h_soc, driver.h_soc.conj().T, atol=1e-12)
        np.testing.assert_allclose((driver.e - driver.e.min()).real * HARTREE2WAVENUMBER, [
            0.0, 45887.18304976328, 45887.18432227726, 45887.189989714265, 51599.47400884429,
        ], atol=1e-5, rtol=0)
        assert_allclose_up_to_sign(self, driver.get_block(1, 0) * HARTREE2WAVENUMBER, np.array([
            [0.4709967390252593 - 6.694129899974075j],
            [0.0 - 14.1984224305132j],
            [0.4709967390252593 + 6.694129899974075j],
        ]), 1e-8)


if __name__ == '__main__':
    print('Full SOC tests for spin-flip TDDFT')
    unittest.main()
