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
        # Converge amplitudes more tightly than the observable assertions.
        # The default residual tolerance (1e-5) leaves dipole outer products
        # sensitive to the eigensolver's trial-space ordering. Retain small
        # independent corrections when requesting the tighter residual.
        td = mf.NTTDA().set(deltaS=-1, nstates=4, conv_tol=1e-9, lindep=1e-18).run()
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
        td = mf.NTTDA().set(deltaS=0, nstates=4, conv_tol=1e-9, lindep=1e-18).run()
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
        td = mf.NTTDA().set(deltaS=1, nstates=4, conv_tol=1e-9, lindep=1e-18).run()
        self.assertTrue(np.all(td.converged))
        dip = td.transition_dipole()
        dip_outer = np.einsum('nx,ny->nxy', dip.conj(), dip)
        ref_outer = np.einsum('nx,ny->nxy', ref_dip.conj(), ref_dip)
        self.assertAlmostEqual(abs(dip_outer - ref_outer).max(), 0, delta=1e-5)
        self.assertAlmostEqual(abs(td.oscillator_strength() - ref_f).max(), 0, delta=1e-5)


    def test_frozen_orbitals_oscillator_strength(self):
        mf = self.mol.ROKS(xc='SVWN').run()
        self.assertTrue(mf.converged)
        # Pure core freezing, then core plus the last virtual orbital.
        references = (
            (2, {
                -1: ([-0.134831604188, 0.085632546917, -0.153508111461], 0.007680813807),
                0: ([1.540800197229, -0.946819349780, 0.037110894550], 0.067491776371),
                1: ([-1.491359659990, 0.935150207487, 0.205502639930], 0.109987901440),
            }),
            ([0, 1, -1], {
                -1: ([-0.134578869750, 0.085519461348, -0.156728216036], 0.007824907757),
                0: ([1.541428916877, -0.946666781738, 0.037001028322], 0.067479224058),
                1: ([1.491143945230, -0.934583572680, -0.205520471385], 0.109958440239),
            }),
        )
        for frozen, refs_by_delta_s in references:
            for delta_s, (ref_dip, ref_f) in refs_by_delta_s.items():
                with self.subTest(frozen=frozen, deltaS=delta_s):
                    td = mf.NTTDA(frozen=frozen).set(
                        deltaS=delta_s, nstates=2,
                    ).run()
                    self.assertTrue(np.all(td.converged))
                    dip = td.transition_dipole()[0]
                    # Compare outer products to allow an overall state phase.
                    np.testing.assert_allclose(
                        np.outer(dip.conj(), dip), np.outer(ref_dip, ref_dip),
                        atol=5e-5, rtol=0,
                    )
                    self.assertAlmostEqual(td.oscillator_strength()[0], ref_f, delta=1e-5)


if __name__ == '__main__':
    print('Full tests for NTTDA transition dipoles and oscillator strengths')
    unittest.main()
