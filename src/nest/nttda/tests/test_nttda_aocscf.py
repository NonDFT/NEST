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
from pyscf import dft, gto, lib
from pyscf.scf import hf
from nest import aocscf, nttda
from nest.nttda import NTTDA


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


class AOCSCFReference(unittest.TestCase):
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
        mf = self.lithium_hydride_cation().ROKS(xc=xc).average_occ()
        mf.conv_tol = 1e-12
        mf.conv_tol_grad = 1e-9
        mf.max_cycle = 100
        mf.verbose = 0
        mf.grids.level = 0
        mf.kernel()
        self.assertTrue(mf.converged)
        return mf

    def test_fixed_occupations_define_a_spin_unpolarized_reference(self):
        mf = self.make_reference()
        np.testing.assert_array_equal(mf.mo_occ, [2, 1, 0, 0, 0, 0])
        self.assertAlmostEqual(mf.mo_occ.sum(), mf.mol.nelectron)

        mo = np.asarray(mf.mo_coeff)
        dm = (mo * np.asarray(mf.mo_occ)) @ mo.conj().T
        veff = mf.get_veff()
        charge_veff = lib.view(mf, dft.rks.RKS).get_veff(dm=dm)
        np.testing.assert_allclose(veff[0], veff[1], atol=0, rtol=0)
        np.testing.assert_allclose(veff[0], charge_veff, atol=1e-12, rtol=0)
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

    def test_nttda_energy_is_independent_of_the_nobeta_flag(self):
        mf = self.make_reference()
        energies = []
        for nobeta in (False, True):
            tdobj = NTTDA(mf).set(
                deltaS=0,
                nobeta=nobeta,
                nstates=2,
                conv_tol=1e-8,
                lindep=1e-18,
                max_cycle=200,
                verbose=0,
            ).run()
            self.assertTrue(np.all(tdobj.converged))
            self.assertTrue(np.all(np.isfinite(tdobj.e)))
            energies.append(tdobj.e)
        np.testing.assert_allclose(energies[0], energies[1], atol=1e-12, rtol=0)

    def test_reference_energy_is_the_high_spin_roks_energy(self):
        mf = self.make_reference()
        high_spin = lib.view(mf, dft.roks.ROKS)
        self.assertAlmostEqual(mf.e_tot, high_spin.energy_tot(), places=12)
        self.assertAlmostEqual(mf.e_avg_occ, mf.energy_tot(), places=12)

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
        mf = mol.ROKS(xc="SVWN").average_occ()
        mf.conv_tol = 1e-12
        mf.max_cycle = 150
        mf.verbose = 0
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
                    lindep=1e-18,
                    max_cycle=200,
                    verbose=0,
                ).run()
                self.assertTrue(np.all(tdobj.converged))
                np.testing.assert_allclose(
                    tdobj.e, reference, atol=2e-6, rtol=0,
                )


if __name__ == "__main__":
    unittest.main()
