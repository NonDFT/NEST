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

import unittest
import numpy as np
from pyscf import gto, lib

from nest.nttda.nttda import _dipole_integrals
from nest.nttda.rks import transition_properties
from nest.nttda.tests.test_rks_transition import make_problem


class Sfx2cTransitionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mol = gto.M(atom='Ne .13 -.21 .37', basis='cc-pvdz', verbose=0)
        cls.mf = cls.mol.RKS().sfx2c1e()

    @classmethod
    def tearDownClass(cls):
        del cls.mf, cls.mol

    def test_length_matches_pyscf_and_origin_shift(self):
        mf = self.mf
        overlap = mf.get_ovlp()
        values, vectors = np.linalg.eigh(overlap)
        coeff = vectors / np.sqrt(values)
        dm = 2 * coeff[:, :5] @ coeff[:, :5].T
        operator = _dipole_integrals(mf, 'length', origin=np.zeros(3))
        expected = mf.dip_moment(dm=dm, unit='AU', verbose=0)
        actual = mf.mol.atom_charges() @ mf.mol.atom_coords() - np.einsum('xpq,qp->x', operator, dm)
        np.testing.assert_allclose(actual, expected, atol=2e-13, rtol=0)
        shift = np.array([.4, -.1, .7])
        moved = _dipole_integrals(mf, 'length', origin=shift)
        np.testing.assert_allclose(moved, operator - shift[:, None, None] * overlap, atol=1e-13, rtol=0)
        bare = _dipole_integrals(mf, 'length', picture_change=False)
        np.testing.assert_allclose(bare, mf.mol.intor_symmetric('int1e_r', comp=3), atol=1e-14, rtol=0)

    def test_velocity_against_explicit_dirac_pauli_blocks(self):
        helper = self.mf.with_x2c
        xmol, contraction = helper.get_xmol()
        n = xmol.nao
        x = helper.get_xmat()
        r = helper._get_rmat(x)
        momentum = 1j * xmol.intor('int1e_ipovlp', comp=3)
        sigma = np.array([[[0, 1], [1, 0]], [[0, -1j], [1j, 0]], [[1, 0], [0, -1]]])
        # Explicit spinor RKB small basis is (sigma.p)/(2c) chi. Build
        # c*alpha in that basis, decouple with scalar X,R, then spin trace.
        u_plus = np.vstack((np.eye(2 * n), np.kron(np.eye(2), x))) @ np.kron(np.eye(2), r)
        expected = []
        for k in range(3):
            ls = sum(np.kron(sigma[k] @ sigma[j], momentum[j]) for j in range(3)) / 2
            four_component = np.zeros((4 * n, 4 * n), dtype=complex)
            four_component[:2 * n, 2 * n:] = ls
            four_component[2 * n:, :2 * n] = ls.conj().T
            transformed = u_plus.conj().T @ four_component @ u_plus
            scalar = (transformed[:n, :n] + transformed[n:, n:]) / 2
            expected.append(1j * contraction.T @ scalar @ contraction)
        actual = _dipole_integrals(self.mf, 'velocity')
        np.testing.assert_allclose(actual, expected, atol=2e-13, rtol=0)
        np.testing.assert_allclose(actual, -actual.transpose(0, 2, 1), atol=1e-13, rtol=0)

    def test_nonrelativistic_limit(self):
        bare = _dipole_integrals(self.mf, 'velocity', picture_change=False)
        errors = []
        for c in (1000., 10000.):
            with lib.light_speed(c):
                errors.append(np.max(abs(_dipole_integrals(self.mf, 'velocity') - bare)))
        self.assertLess(errors[1], 1e-6)
        self.assertLess(errors[1] / errors[0], .011)

    def test_independent_orbitals_x2c_properties(self):
        ground, td = make_problem()
        ground = ground.sfx2c1e()
        td._scf = td._scf.sfx2c1e()
        original = transition_properties(td, ground, gauge='velocity')
        shifted = transition_properties(td, ground, origin=[.4, -.2, .7], gauge='velocity')
        for key in ('overlap', 'dipole', 'velocity_dipole', 'oscillator_strength'):
            np.testing.assert_allclose(shifted[key], original[key], atol=2e-13, rtol=0)
        # Disabling picture change is an explicit bare-operator diagnostic.
        bare = transition_properties(td, ground, gauge='velocity', picture_change=False)
        ground = ground.undo_x2c()
        td._scf = td._scf.undo_x2c()
        nr = transition_properties(td, ground, gauge='velocity')
        for key in bare:
            np.testing.assert_allclose(bare[key], nr[key], atol=2e-14, rtol=0)

    def test_disabled_x2c_helper_is_nonrelativistic(self):
        ground, td = make_problem()
        expected = transition_properties(td, ground, gauge='velocity')
        old_length = td.transition_dipole()
        old_velocity = td.transition_velocity_dipole()
        ground = ground.sfx2c1e()
        ground.with_x2c = False
        # A disabled X2C helper and an ordinary NR object share a representation.
        actual = transition_properties(td, ground, gauge='velocity')
        for key in expected:
            np.testing.assert_allclose(actual[key], expected[key], atol=2e-14, rtol=0)
        td._scf = td._scf.sfx2c1e()
        td._scf.with_x2c = False
        np.testing.assert_allclose(td.transition_dipole(), old_length, atol=2e-14, rtol=0)
        np.testing.assert_allclose(td.transition_velocity_dipole(), old_velocity, atol=2e-14, rtol=0)


if __name__ == '__main__':
    unittest.main()
