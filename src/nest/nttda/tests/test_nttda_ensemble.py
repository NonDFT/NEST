#!/usr/bin/env python
"""Acceptance tests for EnsembleRKS-based NTTDA energies."""

import unittest

import numpy as np

from pyscf import gto
from pyscf.scf import hf
from nest.ensemble_rks import EnsembleRKS
from nest.nttda import NTTDA


class EnsembleRKSReference(unittest.TestCase):
    @staticmethod
    def lithium_hydride_cation():
        return gto.M(
            atom="Li 0 0 0; H 0 0 3.0",
            basis="sto-3g",
            charge=1,
            spin=1,
            unit="Bohr",
            verbose=0,
        )

    def make_reference(self, xc="SVWN"):
        mf = EnsembleRKS(self.lithium_hydride_cation()).set(
            xc=xc,
            conv_tol=1e-12,
            conv_tol_grad=1e-9,
            max_cycle=100,
            verbose=0,
        )
        mf.grids.level = 0
        mf.kernel()
        self.assertTrue(mf.converged)
        return mf

    def test_fixed_occupations_define_a_spin_unpolarized_ensemble(self):
        mf = self.make_reference()
        np.testing.assert_array_equal(mf.mo_occ, [2, 1, 0, 0, 0, 0])
        self.assertAlmostEqual(mf.mo_occ.sum(), mf.mol.nelectron)

        dm = mf.make_rdm1()
        dma, dmb = mf.make_rdm1s()
        np.testing.assert_allclose(dma, dmb, atol=0, rtol=0)
        np.testing.assert_allclose(dma + dmb, dm, atol=1e-14, rtol=0)
        self.assertAlmostEqual(
            np.einsum("ij,ji", dm, mf.get_ovlp()),
            mf.mol.nelectron,
            places=10,
        )

        rng = np.random.default_rng(8)
        fock = rng.standard_normal(dm.shape)
        fock = fock + fock.T
        fock_mo = mf.mo_coeff.T @ fock @ mf.mo_coeff
        unique = hf.uniq_var_indices(mf.mo_occ)
        occupation_difference = mf.mo_occ[None, :] - mf.mo_occ[:, None]
        expected = (fock_mo * occupation_difference)[unique]
        np.testing.assert_allclose(
            mf.get_grad(mf.mo_coeff, mf.mo_occ, fock),
            expected,
            atol=1e-14,
            rtol=0,
        )

    def test_explicit_open_shell_count_must_match_molecular_spin(self):
        with self.assertRaisesRegex(ValueError, "nopen.*mol.spin"):
            EnsembleRKS(self.lithium_hydride_cation(), nopen=3)

    def test_nttda_energy_uses_the_ensemble_fock_for_both_nobeta_values(self):
        mf = self.make_reference()
        energies = []
        for nobeta in (False, True):
            tdobj = NTTDA(mf).set(
                deltaS=0,
                nobeta=nobeta,
                nstates=2,
                conv_tol=1e-8,
                max_cycle=200,
                verbose=0,
            ).run()
            self.assertTrue(np.all(tdobj.converged))
            self.assertTrue(np.all(np.isfinite(tdobj.e)))
            energies.append(tdobj.e)
        np.testing.assert_allclose(energies[0], energies[1], atol=1e-12, rtol=0)

    def test_reference_energy_is_the_stationary_average_occupation_energy(self):
        mf = self.make_reference()
        self.assertEqual(
            mf.reference_energy_semantics,
            "average_occupation_ensemble_rks_energy",
        )
        self.assertTrue(mf.reference_energy_stationary)
        self.assertAlmostEqual(mf.reference_energy(), mf.e_tot, places=14)

    def test_nttda_total_energies_use_the_selected_reference_energy(self):
        mf = self.make_reference()
        tdobj = NTTDA(mf).set(
            deltaS=0,
            nstates=2,
            conv_tol=1e-8,
            max_cycle=200,
            verbose=0,
        ).run()

        self.assertAlmostEqual(tdobj.reference_energy(), mf.e_tot)
        np.testing.assert_allclose(
            tdobj.total_energies(),
            mf.e_tot + tdobj.e,
            atol=1e-13,
            rtol=0,
        )

    def test_nttda_supports_common_functional_families(self):
        for xc in ("HF", "PBE", "TPSS", "M06-2X", "CAM-B3LYP"):
            with self.subTest(xc=xc):
                mf = self.make_reference(xc)
                tdobj = NTTDA(mf).set(
                    deltaS=0,
                    nstates=2,
                    conv_tol=1e-6,
                    max_cycle=200,
                    verbose=0,
                ).run()
                self.assertTrue(np.all(tdobj.converged))
                self.assertTrue(np.all(np.isfinite(tdobj.e)))

    def test_all_three_spin_channels(self):
        mol = gto.M(
            atom="C 0 0 0; H 0 0 2.0; H 0 1.7 -0.5",
            basis="sto-3g",
            spin=2,
            unit="Bohr",
            verbose=0,
        )
        mf = EnsembleRKS(mol).set(
            xc="SVWN",
            conv_tol=1e-12,
            max_cycle=150,
            verbose=0,
        )
        mf.grids.level = 0
        mf.kernel()
        self.assertTrue(mf.converged)

        references = {
            -1: [0.01477165973875402, 0.06753726182098739],
            0: [-0.002622144798075737, 0.2393673138366825],
            1: [0.6745475947851435, 0.8387186686911036],
        }
        for delta_s, reference in references.items():
            with self.subTest(deltaS=delta_s):
                tdobj = NTTDA(mf).set(
                    deltaS=delta_s,
                    nstates=2,
                    conv_tol=1e-7,
                    max_cycle=200,
                    verbose=0,
                ).run()
                self.assertTrue(np.all(tdobj.converged))
                np.testing.assert_allclose(
                    tdobj.e, reference, atol=2e-6, rtol=0,
                )


if __name__ == "__main__":
    unittest.main()
