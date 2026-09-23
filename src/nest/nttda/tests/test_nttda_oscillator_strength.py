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
from nest import nttda


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

    def test_svwn_nttda_oscillator_strength(self):
        mf = self.mol.ROKS(xc='SVWN').run()
        # Dipole outer products allow a whole-state phase while retaining
        # relative Cartesian signs. References were checked independently
        # against the E_pq formulas.

        ref_dip = np.array([
            [-0.134829024677, 0.085633060905, -0.153527653205],
            [0.415641998692, -0.220349792787, -0.018853602643],
            [-0.068297131072, -0.365336383706, 0.031212437760],
        ])
        ref_f = np.array([0.007681628385, 0.037787591904, 0.026266335987])
        td = mf.NTTDA().set(deltaS=-1, nstates=4).run()
        self.assertTrue(np.all(td.converged))
        dip = td.transition_dipole()
        dip_outer = np.einsum('nx,ny->nxy', dip.conj(), dip)
        ref_outer = np.einsum('nx,ny->nxy', ref_dip.conj(), ref_dip)
        self.assertAlmostEqual(abs(dip_outer - ref_outer).max(), 0, delta=1e-5)
        self.assertAlmostEqual(abs(td.oscillator_strength() - ref_f).max(), 0, delta=1e-5)

        ref_dip = np.array([
            [-1.540794701274, 0.946787637437, -0.037112374148],
            [-0.326581746454, -0.158024340978, -0.004595388669],
            [-0.056333525366, 0.071326673082, -0.457181696110],
        ])
        ref_f = np.array([0.067496526009, 0.003926438694, 0.012770953752])
        td = mf.NTTDA().set(deltaS=0, nstates=4).run()
        self.assertTrue(np.all(td.converged))
        dip = td.transition_dipole()
        dip_outer = np.einsum('nx,ny->nxy', dip.conj(), dip)
        ref_outer = np.einsum('nx,ny->nxy', ref_dip.conj(), ref_dip)
        self.assertAlmostEqual(abs(dip_outer - ref_outer).max(), 0, delta=1e-5)
        self.assertAlmostEqual(abs(td.oscillator_strength() - ref_f).max(), 0, delta=1e-5)

        ref_dip = np.array([
            [-1.491358069799, 0.935155153987, 0.205503201827],
            [0.049294564555, 0.012149115879, -0.067379539837],
            [-1.078796933744, -0.328635627721, -0.013042401171],
        ])
        ref_f = np.array([0.109988279396, 0.000658252112, 0.158300119889])
        td = mf.NTTDA().set(deltaS=1, nstates=4).run()
        self.assertTrue(np.all(td.converged))
        dip = td.transition_dipole()
        dip_outer = np.einsum('nx,ny->nxy', dip.conj(), dip)
        ref_outer = np.einsum('nx,ny->nxy', ref_dip.conj(), ref_dip)
        self.assertAlmostEqual(abs(dip_outer - ref_outer).max(), 0, delta=1e-5)
        self.assertAlmostEqual(abs(td.oscillator_strength() - ref_f).max(), 0, delta=1e-5)


if __name__ == '__main__':
    print('Full tests for NTTDA transition dipoles and oscillator strengths')
    unittest.main()
