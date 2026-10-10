# Copyright 2026 The NEST Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

import unittest
import numpy as np
from pyscf import gto
from pyscf.data.nist import HARTREE2WAVENUMBER
from nest import sftda
from nest.soc.soc import clebsch_gordan_rank1


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

    def test_rank_one_clebsch_gordan(self):
        self.assertAlmostEqual(clebsch_gordan_rank1(0, 0, 1, 1, 1), 1.0)
        self.assertAlmostEqual(clebsch_gordan_rank1(1, 1, 0, 1, 1), 2 ** -0.5)
        self.assertAlmostEqual(clebsch_gordan_rank1(1, 0, 1, 1, 1), -2 ** -0.5)
        self.assertAlmostEqual(clebsch_gordan_rank1(1, 0, 0, 0, 0), -3 ** -0.5)

    def test_roks_sftda_soc(self):
        mf = self.mol.ROKS(xc='SVWN').run()
        td = sftda.TDA_SF(mf).set(
            extype=1, collinear='mcol', collinear_samples=50, nstates=3,
        ).run()
        driver = td.SOC(soctype='SOMF')
        driver.kernel()

        self.assertTrue(mf.converged)
        self.assertTrue(np.all(td.converged))
        self.assertAlmostEqual(mf.e_tot, -150.18173594947896, delta=1e-9)
        np.testing.assert_allclose(td.e, [
            -0.2104295981711506, -0.0007174487394460, 0.0251523165536107,
        ], atol=1e-8, rtol=0)
        np.testing.assert_allclose(td.spin_square(), [
            0.0010848962999618905, 1.9999490812871423, 0.031289468589663194,
        ], atol=1e-8, rtol=0)
        self.assertEqual(driver.h_soc.shape, (5, 5))
        np.testing.assert_allclose(driver.h_soc, driver.h_soc.conj().T, atol=1e-12)
        np.testing.assert_allclose((driver.e - driver.e.min()).real * HARTREE2WAVENUMBER, [
            0.0, 46026.50192719676, 46026.502933180134, 46026.50860068575, 51704.26176254972,
        ], atol=1e-5, rtol=0)
        assert_allclose_up_to_sign(self, driver.get_block(1, 0) * HARTREE2WAVENUMBER, np.array([
            [0.4674642361078794 - 6.685128880147436j],
            [0.0 - 14.141141940607179j],
            [0.4674642361078794 + 6.685128880147436j],
        ]), 1e-8)

    def test_uks_sftda_soc(self):
        mf = self.mol.UKS(xc='SVWN').run()
        td = sftda.TDA_SF(mf).set(
            extype=1, collinear='mcol', collinear_samples=50, nstates=3,
        ).run()
        driver = td.SOC(soctype='SOMF')
        driver.kernel()

        self.assertTrue(mf.converged)
        self.assertTrue(np.all(td.converged))
        self.assertAlmostEqual(mf.e_tot, -150.18252681880003, delta=1e-9)
        np.testing.assert_allclose(td.e, [
            -0.2087681123003969, 0.0008054142507056, 0.0266315014304553,
        ], atol=1e-8, rtol=0)
        np.testing.assert_allclose(td.spin_square(), [
            0.0026539116310360, 2.0039691808012283, 0.0372918876839510,
        ], atol=1e-8, rtol=0)
        self.assertEqual(driver.h_soc.shape, (5, 5))
        np.testing.assert_allclose(driver.h_soc, driver.h_soc.conj().T, atol=1e-12)
        np.testing.assert_allclose((driver.e - driver.e.min()).real * HARTREE2WAVENUMBER, [
            0.0, 45996.07760059907, 45996.07872056033, 45996.08436902181, 51664.25143819543,
        ], atol=1e-5, rtol=0)
        assert_allclose_up_to_sign(self, driver.get_block(1, 0) * HARTREE2WAVENUMBER, np.array([
            [-0.4695042701337793 + 6.699202067333119j],
            [0.0 + 14.109500425297364j],
            [-0.4695042701337793 - 6.699202067333119j],
        ]), 1e-8)


if __name__ == '__main__':
    print('Full SOC tests for spin-flip TDA')
    unittest.main()
