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
from pyscf import gto
from nest import sftda


class KnownValues(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        mol = gto.Mole()
        mol.verbose = 0
        mol.output = '/dev/null'
        mol.atom = """
        O  0.64372820  0.14077399 -0.04477253
        O -0.64862595 -0.12779073 -0.05445498
        H  1.16027512 -0.65947800  0.36730132
        H -1.12109306  0.55561188  0.42651873
        """
        mol.spin = 2
        mol.basis = '631g'
        cls.mol = mol.build()

    @classmethod
    def tearDownClass(cls):
        cls.mol.stdout.close()

    def test_svwn_sftda_oscillator_strength(self):
        mf = self.mol.UKS(xc='SVWN').run()
        # Dipole outer products allow a whole-state phase while retaining
        # relative Cartesian signs. References were checked independently
        # using the one-electron operator in the determinant basis.

        ref_dip = np.array([
            [0.003635607826, -0.003067882348, 0.007467093849],
            [0.135384808571, -0.085800520970, 0.123668401075],
            [-0.286961472890, 0.153271168849, 0.014502065364],
        ])
        ref_f = np.array([0.000010951899, 0.006431847063, 0.017443630663])
        td = mf.TDA_SF().set(nstates=4).run()
        self.assertTrue(np.all(td.converged))
        dip = td.transition_dipole()
        dip_outer = np.einsum('nx,ny->nxy', dip.conj(), dip)
        ref_outer = np.einsum('nx,ny->nxy', ref_dip.conj(), ref_dip)
        self.assertAlmostEqual(abs(dip_outer - ref_outer).max(), 0, delta=1e-5)
        self.assertAlmostEqual(abs(td.oscillator_strength() - ref_f).max(), 0, delta=1e-5)

    def test_svwn_sftddft_oscillator_strength(self):
        mf = self.mol.UKS(xc='SVWN').run()
        # References include both X and Y contributions, independently checked
        # by contracting the occupied/virtual blocks with MO dipole integrals.
        ref_dip = np.array([
            [0.003507083925, -0.002819963562, 0.006121995290],
            [-0.138088055687, 0.087342330714, -0.125251403979],
            [0.287610742086, -0.153386582531, -0.015315643773],
        ])
        ref_f = np.array([0.000008046577, 0.006643398920, 0.017510757397])
        td = mf.TDDFT_SF().set(nstates=4).run()
        self.assertTrue(np.all(td.converged))
        dip = td.transition_dipole()
        dip_outer = np.einsum('nx,ny->nxy', dip.conj(), dip)
        ref_outer = np.einsum('nx,ny->nxy', ref_dip.conj(), ref_dip)
        self.assertAlmostEqual(abs(dip_outer - ref_outer).max(), 0, delta=1e-5)
        self.assertAlmostEqual(abs(td.oscillator_strength() - ref_f).max(), 0, delta=1e-5)


if __name__ == '__main__':
    print('Full tests for SFTDA/SFTDDFT transition dipoles and oscillator strengths')
    unittest.main()
