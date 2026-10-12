# Copyright 2026 The NEST Developers. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Algebra tests using direct minors and PySCF's full-CI one-body operator."""

from itertools import combinations
import unittest
import warnings

import numpy as np
from pyscf import fci, gto

from nest.nttda import NTTDA, rks
from nest.nttda.nttda import transition_dm, _orbital_indices


def direct_element(matrix, operator):
    # Deliberately slow oracle: explicit minors, including singular M.
    cofactor = np.empty_like(matrix)
    for i in range(len(matrix)):
        for j in range(len(matrix)):
            minor = np.delete(np.delete(matrix, i, axis=0), j, axis=1)
            cofactor[i, j] = (-1) ** (i + j) * np.linalg.det(minor)
    return np.linalg.det(matrix), np.einsum('ij,xij->x', cofactor, operator)


def make_problem(charge=0):
    # Synthetic orthonormal orbitals and normalized singlet amplitudes isolate
    # matrix-element algebra from SCF/eigensolver convergence. No SCF is run.
    mol = gto.M(atom='H 0 0 0; H 0.2 0 1.3; H 1.4 0.1 0; H 1.1 0.3 1.5',
                basis='sto-3g', charge=charge, verbose=0)
    ground = mol.RKS()
    triplet_mol = mol.copy()
    triplet_mol.spin = 2
    mf = triplet_mol.ROKS()
    s = ground.get_ovlp()
    values, vectors = np.linalg.eigh(s)
    mf.mo_coeff = (vectors / np.sqrt(values)) @ vectors.T
    rng = np.random.default_rng(17)
    rotation = np.linalg.qr(rng.normal(size=s.shape))[0]
    ground.mo_coeff = mf.mo_coeff @ rotation
    nocc = mol.nelectron // 2
    ground.mo_occ = np.array([2] * nocc + [0] * (mol.nao - nocc))
    mf.mo_occ = np.zeros(mol.nao)
    mf.mo_occ[[0, 2]] = 1
    if nocc == 2:
        mf.mo_occ[1] = 2  # C/O/V are deliberately non-contiguous.
    ground.e_tot, mf.e_tot = -2.0, -1.5
    td = NTTDA(mf)
    td.deltaS = -1
    cs, os, vs = _orbital_indices(td)
    nc, no, nv = len(cs), len(os), len(vs)
    x = rng.normal(size=(3, nc + no, no + nv))
    for root in x:
        root[nc:, :no] -= np.eye(no) * np.trace(root[nc:, :no]) / no
    x = np.linalg.qr(x.reshape(3, -1).T)[0].T.reshape(x.shape)
    td.xy = [(root, 0) for root in x]
    td.e = np.array([-0.7, 0.2, 0.4])
    return ground, td


def full_ci_states(td):
    nmo = len(td._scf.mo_occ)
    cs, os, vs = _orbital_indices(td)
    strings = list(map(int, fci.cistring.make_strings(range(nmo), len(cs) + 1)))
    address = {s: i for i, s in enumerate(strings)}
    basis = list(rks._singlet_csfs(cs, os, vs, nmo))
    expansion = np.zeros((len(basis), len(strings), len(strings)))
    for row, csf in enumerate(basis):
        for det, coefficient in csf.items():
            alpha, beta = det & ((1 << nmo) - 1), det >> nmo
            expansion[row, address[alpha], address[beta]] = coefficient
    amplitudes = np.array([root.ravel() for root, _ in td.xy])
    return np.einsum('ri,iab->rab', amplitudes, expansion), expansion, strings


class RksTransitionTests(unittest.TestCase):
    def test_column_updates_and_svd_against_minors(self):
        rng = np.random.default_rng(6)
        cross_r = rng.normal(size=(3, 3, 7))  # Non-symmetric operators catch transposes.
        regular = np.linalg.qr(rng.normal(size=(7, 7)))[0][:3]
        same_mos = np.eye(7)[[0, 2, 4]]
        singular_base = np.eye(7)[[1, 3, 5]]
        for cross_s in (regular, same_mos, singular_base):
            filled = [0, 2, 4]
            occupied = [o for o in combinations(range(7), 3) if len(set(o) - set(filled)) <= 2]
            result = rks._spin_elements(cross_s, cross_r, occupied, filled)
            for occ in occupied:
                expected_s, expected_r = direct_element(cross_s[:, occ], cross_r[:, :, occ])
                actual_s, actual_r = result[occ]
                self.assertAlmostEqual(actual_s, expected_s, delta=2e-12)
                np.testing.assert_allclose(actual_r, expected_r, atol=2e-12, rtol=0)
        # A tiny 1x1 auxiliary matrix has cond=1 but must use the SVD path.
        result = rks._spin_elements(np.array([[1e-14, 1.0]]), np.ones((3, 1, 2)), [(0,), (1,)], [0])
        np.testing.assert_allclose(result[(1,)][1], np.ones(3), atol=1e-14)

    def test_svd_singular_one_body_selection_rule(self):
        rng = np.random.default_rng(51)
        for n in (1, 2, 5):
            u = np.linalg.qr(rng.normal(size=(n, n)))[0]
            v = np.linalg.qr(rng.normal(size=(n, n)))[0]
            operator = rng.normal(size=(3, n, n))
            for rank in range(n + 1):
                matrix = (u * np.array([0.7] * rank + [0.0] * (n - rank))) @ v.T
                expected_s, expected_r = direct_element(matrix, operator)
                actual_s, actual_r = rks._svd_element(matrix, operator)
                self.assertAlmostEqual(actual_s, expected_s, delta=1e-12)
                np.testing.assert_allclose(actual_r, expected_r, atol=1e-12, rtol=0)

    def test_csfs_spin_and_existing_nttda_density(self):
        _, td = make_problem()
        ci, expansion, _ = full_ci_states(td)
        flat = expansion.reshape(len(expansion), -1)
        np.testing.assert_allclose(flat @ flat.T, np.eye(len(expansion)), atol=1e-14)
        for i, bra in enumerate(ci):
            self.assertAlmostEqual(fci.spin_op.spin_square0(bra, 4, (2, 2))[0], 0, delta=1e-13)
            for j, ket in enumerate(ci):
                if i != j:
                    gamma = fci.direct_spin1.trans_rdm1(bra, ket, 4, (2, 2)).T
                    np.testing.assert_allclose(transition_dm(td, i, j), gamma, atol=2e-14, rtol=0)

    def test_independent_orbitals_against_full_ci(self):
        ground, td = make_problem()
        actual = rks.transition_properties(td, ground)
        ci, _, strings = full_ci_states(td)
        ct = td._scf.mo_coeff
        # Expand the independent RKS determinant into the complete NTTDA MO
        # Fock space using exterior products; evaluate r with PySCF FCI.
        transform = ct.T @ ground.get_ovlp() @ ground.mo_coeff[:, ground.mo_occ == 2]
        coefficients = [np.linalg.det(transform[[p for p in range(4) if (s >> p) & 1]]) for s in strings]
        bra = np.outer(coefficients, coefficients)
        dip_mo = np.einsum('up,xuv,vq->xpq', ct, ground.mol.intor_symmetric('int1e_r', comp=3), ct)
        for i, ket in enumerate(ci):
            self.assertAlmostEqual(actual['overlap'][i], np.vdot(bra, ket), delta=2e-14)
            gamma = fci.direct_spin1.trans_rdm1(bra, ket, 4, (2, 2)).T
            np.testing.assert_allclose(actual['electronic_dipole'][i], np.einsum('pq,xpq->x', gamma, dip_mo),
                                       atol=2e-14, rtol=0)
        np.testing.assert_allclose(actual['excitation_energy'], td.e + 0.5, atol=1e-14)
        np.testing.assert_allclose(actual['oscillator_strength'],
                                   (2 / 3) * (td.e + 0.5) * np.sum(actual['dipole']**2, axis=1))
        self.assertLess(actual['oscillator_strength'][0], 0)
        selected = rks.transition_properties(td, ground, state=[3, 1])
        for key in actual:
            np.testing.assert_allclose(selected[key], actual[key][[2, 0]])
        self.assertEqual(rks.transition_properties(td, ground, state=2)['dipole'].shape, (1, 3))
        # An occupied RKS rotation changes each spin determinant's phase by
        # det(U), so the full closed-shell determinant is unchanged.
        ground.mo_coeff[:, :2] = ground.mo_coeff[:, :2] @ np.array([[0.6, 0.8], [0.8, -0.6]])
        rotated = rks.transition_properties(td, ground)
        for key in actual:
            np.testing.assert_allclose(rotated[key], actual[key], atol=2e-14, rtol=0)

    def test_origin_and_charged_warning(self):
        shift = np.array([1.2, -0.7, 0.4])
        for charge in (0, 2):
            ground, td = make_problem(charge)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                original = rks.transition_properties(td, ground)
                moved = rks.transition_properties(td, ground, origin=shift)
            self.assertEqual(len(caught), 0 if charge == 0 else 2)  # Once per call, not per root.
            if charge:
                self.assertIn('charge=2', str(caught[0].message))
                self.assertIn('max|overlap|', str(caught[0].message))
            overlap_shift = original['overlap'][:, None] * shift
            np.testing.assert_allclose(moved['electronic_dipole'],
                                       original['electronic_dipole'] - ground.mol.nelectron * overlap_shift, atol=2e-14)
            np.testing.assert_allclose(moved['dipole'], original['dipole'] + charge * overlap_shift, atol=2e-14)
            if charge == 0:
                np.testing.assert_allclose(moved['oscillator_strength'], original['oscillator_strength'], atol=2e-14)

    def test_unsupported_inputs(self):
        ground, td = make_problem()
        for state in (0, 4, 1.5):
            with self.assertRaises(ValueError):
                rks.transition_properties(td, ground, state=state)
        td.deltaS = 0
        with self.assertRaisesRegex(ValueError, 'deltaS=-1'):
            rks.transition_properties(td, ground)
        td.deltaS = -1
        ground = ground.sfx2c1e()
        with self.assertRaisesRegex(NotImplementedError, 'X2C'):
            rks.transition_properties(td, ground)


if __name__ == '__main__':
    unittest.main()
