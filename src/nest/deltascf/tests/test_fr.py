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
from unittest import mock

import numpy as np
from scipy.linalg import expm
from pyscf import gto, scf, symm
from nest.deltascf import fr


class KnownValues(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mol = gto.M(atom='O 0 0 0; H 0 -.757 .587; H 0 .757 .587',
                        basis='6-31g', spin=2, verbose=0)

    def test_freeze_release(self):
        for xc, use_symmetry in ((None, False), ('BHandHLYP', False),
                                 (None, True), ('BHandHLYP', True)):
            with self.subTest(xc=xc, symmetry=use_symmetry):
                mol = gto.M(atom=self.mol.atom, basis=self.mol.basis, spin=2,
                            symmetry=use_symmetry, verbose=0)
                ground_mol = mol.copy()
                ground_mol.spin = 0
                ground = ground_mol.RHF() if xc is None else ground_mol.RKS(xc=xc)
                mf = mol.ROHF() if xc is None else mol.ROKS(xc=xc)
                if xc is not None:
                    ground.grids.level = mf.grids.level = 0
                ground.kernel()
                occ = ground.mo_occ.copy()
                occ[[4, 5]] = 1
                c = ground.mo_coeff.copy()
                s = mf.get_ovlp()
                stages = []

                def callback(env):
                    stages.append(env['fr_stage'])
                    coeff = env['mo_coeff']
                    np.testing.assert_allclose(coeff.T @ s @ coeff, np.eye(occ.size), atol=1e-10)
                    if env['fr_stage'] == 'freeze':
                        np.testing.assert_array_equal(coeff[:, occ == 1], c[:, occ == 1])
                        free_basis = env['mf_diis'].Corth
                        np.testing.assert_allclose(free_basis.T @ s @ c[:, occ == 1], 0, atol=1e-10)
                        # Independently verify that projected CDIIS residual
                        # and the physical frozen gradient have the same zero.
                        dm_total = env['dm'].sum(axis=0)
                        fock = env['fock']  # Physical Fock at CURRENT density.
                        residual = free_basis.T @ (fock @ dm_total @ s - s @ dm_total @ fock) @ free_basis
                        full_gradient = mf.get_grad(coeff, occ, fock)
                        all_rotations = ((occ == 0)[:, None] & (occ > 0)) | ((occ != 2)[:, None] & (occ == 2))
                        free_entries = ((occ == 0)[:, None] & (occ == 2))[all_rotations]
                        frozen_gradient = full_gradient[free_entries]
                        self.assertAlmostEqual(np.linalg.norm(residual),
                                               np.sqrt(2) * np.linalg.norm(frozen_gradient), places=10)
                        np.testing.assert_allclose(env['mf'].get_grad(coeff, occ, fock), frozen_gradient, atol=1e-12)
                        if stages.count('freeze') == 1:
                            # Energy finite differences are independent of
                            # the gradient packing and the DIIS construction.
                            direction = np.zeros_like(full_gradient)
                            direction[free_entries] = frozen_gradient
                            direction /= np.linalg.norm(direction)
                            rotation = scf.hf.unpack_uniq_var(direction, occ)
                            eps = 1e-4
                            plus = mf.energy_tot(dm=mf.make_rdm1(coeff @ expm(eps * rotation), occ))
                            minus = mf.energy_tot(dm=mf.make_rdm1(coeff @ expm(-eps * rotation), occ))
                            self.assertAlmostEqual((plus-minus)/(2*eps), 2 * full_gradient @ direction, places=6)
                    else:
                        self.assertEqual(np.count_nonzero(env['mo_occ'] == 1), 2)

                opt = mf.FR().set(conv_tol=1e-10, conv_tol_grad=1e-6, max_cycle=100,
                                  callback=callback)
                self.assertIsInstance(opt, fr.FR)
                self.assertTrue(opt.istype('ROHF'))
                # First-order implementation must not require response kernels.
                with mock.patch.object(type(mf), 'gen_response', side_effect=AssertionError('Hessian used')):
                    opt.kernel(c, occ)
                self.assertTrue(opt.freeze_converged)
                self.assertTrue(opt.converged)
                self.assertIn('freeze', stages)
                self.assertIn('release', stages)
                self.assertLess(np.linalg.norm(opt.get_grad(opt.mo_coeff, opt.mo_occ)), 1e-6)
                self.assertAlmostEqual(opt.e_tot, opt.energy_tot(dm=opt.make_rdm1()), places=10)
                np.testing.assert_array_equal(opt.get_occ(opt.mo_energy, opt.mo_coeff), opt.mo_occ)
                # Independent conventional triplet SCF reference for this valence state.
                ref = mf.copy().set(conv_tol=1e-11, conv_tol_grad=1e-7)
                ref.kernel()
                self.assertTrue(ref.converged)
                self.assertAlmostEqual(opt.e_tot, ref.e_tot, places=7)
                self.assertIsNone(mf.mo_coeff)
                self.assertIs(opt.callback, callback)
                self.assertIs(opt.FR(), opt)
                if use_symmetry:
                    initial_symmetry = symm.label_orb_symm(mol, mol.irrep_id, mol.symm_orb, c)
                    final_symmetry = symm.label_orb_symm(mol, mol.irrep_id, mol.symm_orb, opt.mo_coeff)
                    np.testing.assert_array_equal(opt.mo_coeff.orbsym, final_symmetry)
                    for irrep in mol.irrep_id:
                        for occupied in (lambda o: o > 0, lambda o: o == 2):
                            self.assertEqual(np.count_nonzero(occupied(occ) & (initial_symmetry == irrep)),
                                             np.count_nonzero(occupied(opt.mo_occ) & (final_symmetry == irrep)))

    def test_failed_freeze_does_not_release(self):
        ground_mol = self.mol.copy()
        ground_mol.spin = 0
        ground = ground_mol.RHF().run()
        occ = ground.mo_occ.copy()
        occ[[0, 5]] = 1
        stages = []
        opt = fr.FR(self.mol.ROHF()).set(freeze_max_cycle=1, freeze_tol=1e-12,
                                       callback=lambda env: stages.append(env['fr_stage']))
        with self.assertRaisesRegex(RuntimeError, 'release was not started'):
            opt.kernel(ground.mo_coeff, occ)
        self.assertFalse(opt.converged)
        self.assertFalse(opt.freeze_converged)
        self.assertEqual(stages, ['freeze'])

    def test_reject_invalid_reference(self):
        mf = self.mol.ROHF().run()
        with self.assertRaisesRegex(ValueError, 'S-orthonormal'):
            mf.FR().kernel(mf.mo_coeff * 2, mf.mo_occ)
        with self.assertRaisesRegex(ValueError, 'two SOMOs'):
            mf.FR().kernel(mf.mo_coeff, np.zeros_like(mf.mo_occ))
        with self.assertRaises(NotImplementedError):
            fr.FR(self.mol.UHF())

    def test_symmetry_finalization_reorders_columns(self):
        mol = gto.M(atom=self.mol.atom, basis=self.mol.basis, spin=2, symmetry=True, verbose=0)
        mf = mol.ROHF().set(conv_tol=1e-11, conv_tol_grad=1e-8).run()
        order = np.arange(mf.mo_occ.size)[::-1]
        # In particular, singly occupied columns are no longer immediately
        # after doubly occupied ones. PySCF finalization will reorder them.
        coeff = np.asarray(mf.mo_coeff[:, order])
        occupations = mf.mo_occ[order]
        opt = mf.FR().set(conv_tol=1e-10, conv_tol_grad=1e-7)
        opt.kernel(coeff, occupations)
        self.assertTrue(opt.converged)
        self.assertAlmostEqual(opt.e_tot, mf.e_tot, places=9)
        np.testing.assert_allclose(opt.make_rdm1(), mf.make_rdm1(), atol=1e-6)
        np.testing.assert_array_equal(opt.get_occ(opt.mo_energy, opt.mo_coeff), opt.mo_occ)

    def test_symmetry_rejects_mixed_orbitals_and_conflicting_counts(self):
        mol = gto.M(atom=self.mol.atom, basis=self.mol.basis, spin=2, symmetry=True, verbose=0)
        mf = mol.ROHF().run()
        labels = mf.mo_coeff.orbsym
        first = 0
        second = np.flatnonzero(labels != labels[first])[0]
        mixed = mf.mo_coeff.copy()
        mixed[:, first] = (mf.mo_coeff[:, first] + mf.mo_coeff[:, second]) / np.sqrt(2)
        mixed[:, second] = (mf.mo_coeff[:, first] - mf.mo_coeff[:, second]) / np.sqrt(2)
        with self.assertRaisesRegex(ValueError, 'not symmetrized'):
            mf.FR().kernel(mixed, mf.mo_occ)
        name = mol.irrep_name[mol.irrep_id.index(int(labels[first]))]
        mf.irrep_nelec = {name: (0, 0)}
        with self.assertRaisesRegex(ValueError, 'conflict'):
            mf.FR().kernel(mf.mo_coeff, mf.mo_occ)

    def test_imom_rejects_nonnested_spaces(self):
        mf = self.mol.ROHF().run()
        setocc = np.asarray((mf.mo_occ > 0, mf.mo_occ == 2), dtype=float)

        def invalid_imom(mf, coeff, occupations):
            invalid = np.zeros_like(mf.mo_occ)
            invalid[:3] = 2
            invalid[3:7] = 1  # Same electron count, but four SOMOs.
            mf.get_occ = lambda *args: invalid

        with mock.patch.object(scf.addons, 'mom_occ', side_effect=invalid_imom):
            fr._set_imom(mf, mf.mo_coeff, setocc)
        with self.assertRaisesRegex(RuntimeError, 'nonnested'):
            mf.get_occ(mf.mo_energy, mf.mo_coeff)


if __name__ == '__main__':
    unittest.main()
