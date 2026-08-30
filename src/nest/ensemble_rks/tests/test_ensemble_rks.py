import unittest

import numpy as np
from pyscf import gto

from nest.ensemble_rks import EnsembleRKS
from nest.nttda import NTTDA


class EnsembleRKSReferenceTest(unittest.TestCase):
    def make_reference(self):
        mol = gto.M(
            atom='''
                O   0.64372820   0.14077399  -0.04477253
                O  -0.64862595  -0.12779073  -0.05445498
                H   1.16027512  -0.65947800   0.36730132
                H  -1.12109306   0.55561188   0.42651873
            ''',
            basis='6-31g',
            spin=2,
            unit='Angstrom',
            verbose=0,
        )
        mf = EnsembleRKS(mol, xc='SVWN')
        mf.conv_tol = 1e-12
        mf.conv_tol_grad = 1e-9
        mf.max_cycle = 100
        mf.grids.level = 3
        mf.grids.prune = None
        mf.small_rho_cutoff = 0.0
        mf.kernel()
        self.assertTrue(mf.converged)
        return mf

    def test_average_occupation_reference_energy_is_used_for_total_energies(self):
        mf = self.make_reference()
        self.assertEqual(np.count_nonzero(mf.mo_occ == 1), mf.nopen)
        self.assertEqual(np.count_nonzero(mf.mo_occ == 2), mf.nclosed)
        self.assertAlmostEqual(mf.reference_energy(), mf.e_tot, places=14)

        td = NTTDA(mf).set(deltaS=0, nstates=2, conv_tol=1e-5, max_cycle=200, verbose=0)
        td.kernel()

        self.assertTrue(np.all(td.converged))
        np.testing.assert_allclose(
            td.total_energies(),
            mf.reference_energy() + td.e,
            atol=1e-13,
            rtol=0,
        )

    def test_nobeta_does_not_change_an_ensemble_reference(self):
        mf = self.make_reference()
        roots = []
        for nobeta in (False, True):
            td = NTTDA(mf).set(
                deltaS=0,
                nobeta=nobeta,
                nstates=2,
                conv_tol=1e-5,
                max_cycle=200,
                verbose=0,
            )
            td.kernel()
            self.assertTrue(np.all(td.converged))
            roots.append(td.e)
        np.testing.assert_allclose(roots[0], roots[1], atol=1e-12, rtol=0)


if __name__ == '__main__':
    unittest.main()
