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
from pyscf import dft, gto, scf

from nest.soc import SOTDDFT, TDRHFSOC


class KnownValues(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mol = gto.M(
            atom='O 0 0 0; H 0 -0.757 0.587; H 0 0.757 0.587',
            basis='sto-3g',
            verbose=0,
        )
        cls.mf = scf.RHF(cls.mol).run(conv_tol=1e-12)
        cls.nocc = np.count_nonzero(cls.mf.mo_occ == 2)
        cls.nvir = np.count_nonzero(cls.mf.mo_occ == 0)
        cls.nov = cls.nocc * cls.nvir

    @staticmethod
    def _dense_matrix(td):
        vind, hdiag = td.gen_vind()
        identity = np.eye(hdiag.size, dtype=np.complex128)
        return vind(identity).T

    def test_vind_matches_full_state_interaction(self):
        singlet = self.mf.TDA().set(singlet=True, nstates=self.nov).run()
        triplet = self.mf.TDA().set(singlet=False, nstates=self.nov).run()
        state_interaction = TDRHFSOC(
            singlet, triplet, soctype='1e', include_reference=True,
        )
        state_interaction.build_hamiltonian()

        direct = SOTDDFT(self.mf, soctype='1e', include_reference=True)
        h_direct = self._dense_matrix(direct)
        dimension = 1 + 4 * self.nov
        self.assertEqual(h_direct.shape, (dimension, dimension))
        np.testing.assert_allclose(h_direct, h_direct.conj().T, atol=1e-12, rtol=0)

        # Transform the direct excitation basis [ref,S,T-1,T0,T+1] into the
        # complete spin-free eigenstate basis used by TDRHFSOC.
        transform = np.zeros((dimension, dimension), dtype=np.complex128)
        transform[0, 0] = 1
        for root, (x, _) in enumerate(singlet.xy):
            amplitude = (x / np.linalg.norm(x)).ravel()
            transform[1:1 + self.nov, 1 + root] = amplitude

        triplet_row = 1 + self.nov
        triplet_col = 1 + self.nov
        for root, (x, _) in enumerate(triplet.xy):
            amplitude = (x / np.linalg.norm(x)).ravel()
            for component in range(3):
                row = triplet_row + component * self.nov
                col = triplet_col + 3 * root + component
                transform[row:row + self.nov, col] = amplitude

        np.testing.assert_allclose(
            transform.conj().T @ transform, np.eye(dimension), atol=1e-12, rtol=0,
        )
        projected = transform.conj().T @ h_direct @ transform
        np.testing.assert_allclose(projected, state_interaction.h_soc, atol=1e-11, rtol=0)

    def test_kernel_and_amplitude_layout(self):
        td = SOTDDFT(self.mf, soctype='1e', include_reference=True).set(nstates=5)
        exact = np.linalg.eigvalsh(self._dense_matrix(td))[:td.nstates]
        energy, xy = td.kernel()

        self.assertTrue(np.all(td.converged))
        np.testing.assert_allclose(energy, exact, atol=1e-8, rtol=0)
        self.assertEqual(len(xy), td.nstates)
        for x, y in xy:
            self.assertEqual(x.shape, (1 + 4 * self.nov,))
            self.assertTrue(np.iscomplexobj(x))
            self.assertEqual(y, 0)

    def test_without_reference(self):
        td = self.mf.SOTDDFT(soctype='1e', include_reference=False)
        self.assertIsInstance(td, SOTDDFT)
        h_direct = self._dense_matrix(td)
        self.assertEqual(h_direct.shape, (4 * self.nov, 4 * self.nov))
        np.testing.assert_allclose(h_direct, h_direct.conj().T, atol=1e-12, rtol=0)

    def test_get_ab_matches_vind_for_random_complex_vector(self):
        rng = np.random.default_rng(12)
        cases = ((False, None), (True, None), (True, 1))
        for include_reference, frozen in cases:
            td = SOTDDFT(
                self.mf, soctype='1e', include_reference=include_reference, frozen=frozen,
            )
            hamiltonian = td.get_ab()
            vector = rng.standard_normal(hamiltonian.shape[0])
            vector = vector + 1j * rng.standard_normal(hamiltonian.shape[0])
            vind, _ = td.gen_vind()
            error = vind(vector)[0] - hamiltonian.dot(vector)

            np.testing.assert_allclose(hamiltonian, hamiltonian.conj().T, atol=1e-12, rtol=0)
            self.assertLess(np.linalg.norm(error), 1e-12)
            self.assertLess(np.max(np.abs(error)), 1e-12)

    def test_rks_response_accepts_complex_trials(self):
        mf = dft.RKS(self.mol, xc='lda,vwn').run(conv_tol=1e-12)
        td = SOTDDFT(mf, soctype='1e', include_reference=True)
        h_direct = self._dense_matrix(td)
        hamiltonian = td.get_ab()
        rng = np.random.default_rng(21)
        vector = rng.standard_normal(hamiltonian.shape[0])
        vector = vector + 1j * rng.standard_normal(hamiltonian.shape[0])
        vind, _ = td.gen_vind()
        error = vind(vector)[0] - hamiltonian.dot(vector)

        self.assertTrue(np.iscomplexobj(h_direct))
        np.testing.assert_allclose(h_direct, h_direct.conj().T, atol=1e-10, rtol=0)
        self.assertLess(np.linalg.norm(error), 1e-12)
        self.assertLess(np.max(np.abs(error)), 1e-12)


if __name__ == '__main__':
    unittest.main()
