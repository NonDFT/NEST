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
from pyscf.data.nist import HARTREE2WAVENUMBER

from nest.soc import SOTDA


class KnownValues(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        mol = gto.M(
            atom='O 0 0 0; H 0 -0.757 0.587; H 0 0.757 0.587',
            basis='sto-3g',
            verbose=0,
        )
        cls.mf = scf.RHF(mol).run(conv_tol=1e-12)

    def test_rhf_1e_soc_spectrum(self):
        td = SOTDA(self.mf, soctype='1e').run(nstates=8)
        self.assertTrue(np.all(td.converged))

        # ORCA 6.1.1: HF/STO-3G, NoFrozenCore, all 10 singlets and
        # 10 triplets, TDA, SOCType 3 / SOCFlags 1,0,0,0 (one-electron SOC).
        # Ground-state stabilization and excitation energies are in cm^-1.
        energy = td.e * HARTREE2WAVENUMBER
        self.assertAlmostEqual(energy[0], -0.2209, delta=0.01)
        np.testing.assert_allclose(energy[1:] - energy[0], [
            89417.55, 89417.55, 89417.73, 106350.94,
            108012.94, 108013.05, 108015.33,
        ], atol=0.01, rtol=0)

    def test_frozen_rks_vind_and_kernel_match_get_ab(self):
        mf = dft.RKS(self.mf.mol, xc='lda,vwn').run()
        td = SOTDA(mf, soctype='1e', frozen=1)
        h = td.get_ab()
        vind, _ = td.gen_vind()
        rng = np.random.default_rng(12)
        x = rng.standard_normal(h.shape[0]) + 1j * rng.standard_normal(h.shape[0])
        np.testing.assert_allclose(vind(x)[0], h @ x, atol=1e-12, rtol=0)

        energies, _ = td.kernel(nstates=5)
        self.assertTrue(np.all(td.converged))
        np.testing.assert_allclose(energies, np.linalg.eigvalsh(h)[:5], atol=1e-6, rtol=0)

    def test_frozen_core_get_ab(self):
        # Same ORCA input with the O 1s frozen: 8 singlets and 8 triplets.
        h = SOTDA(self.mf, soctype='1e', frozen=1).get_ab()
        self.assertEqual(h.shape, (33, 33))
        np.testing.assert_allclose(h, h.conj().T, atol=1e-12, rtol=0)
        energy = np.linalg.eigvalsh(h) * HARTREE2WAVENUMBER
        np.testing.assert_allclose(energy[5:8] - energy[0], [
            108018.31, 108018.41, 108020.68,
        ], atol=0.01, rtol=0)


if __name__ == '__main__':
    unittest.main()
