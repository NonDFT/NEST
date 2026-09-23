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

import io
import unittest
from unittest.mock import patch
import numpy as np
from pyscf import dft, gto
from pyscf.lib import logger
from nest import nttda


class KnownValues(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        mol = gto.Mole()
        mol.verbose = 0
        mol.output = '/dev/null'
        mol.atom = """
        O                  0.64372820    0.14077399   -0.04477253
        O                 -0.64862595   -0.12779073   -0.05445498
        H                  1.16027512   -0.65947800    0.36730132
        H                 -1.12109306    0.55561188    0.42651873
        """
        mol.charge = 0
        mol.spin = 2
        mol.basis = '631g'
        cls.mol = mol.build()

    @classmethod
    def tearDownClass(cls):
        cls.mol.stdout.close()

    def test_hf_nttda(self):
        mf = self.mol.ROKS(xc='HF').run()

        ref = np.array([0.26373033968267973, 0.32114587049263738])
        td = mf.NTTDA().set(nstates=2, deltaS=1, nobeta=True).run()
        self.assertTrue(np.all(td.converged))
        self.assertAlmostEqual(abs(td.e - ref).max(), 0, delta=1e-6)

        ref = np.array([-0.25588162251385815, 0.03179164805915535])
        td = mf.NTTDA().set(nstates=2, deltaS=-1, nobeta=False).run()
        self.assertTrue(np.all(td.converged))
        self.assertAlmostEqual(abs(td.e - ref).max(), 0, delta=1e-6)

        ref = np.array([-0.021227306082554027, 0.03681224565830669])
        td = mf.NTTDA().set(nstates=2, deltaS=0, nobeta=False).run()
        self.assertTrue(np.all(td.converged))
        self.assertAlmostEqual(abs(td.e - ref).max(), 0, delta=1e-6)

    def test_svwn_nttda(self):
        mf = self.mol.ROKS(xc='SVWN').run()

        ref = np.array([-0.21136285952298853, 0.022829192982022128])
        td = mf.NTTDA().set(nstates=2, deltaS=-1, nobeta=True).run()
        self.assertTrue(np.all(td.converged))
        self.assertAlmostEqual(abs(td.e - ref).max(), 0, delta=1e-6)

        ref = np.array([-0.0014224229333087768, 0.029907227771976085])
        td = mf.NTTDA().set(nstates=2, deltaS=0, nobeta=True).run()
        self.assertTrue(np.all(td.converged))
        self.assertAlmostEqual(abs(td.e - ref).max(), 0, delta=1e-6)

        ref = np.array([0.2621305574444208, 0.3146577468311684])
        td = mf.NTTDA().set(nstates=2, deltaS=1, nobeta=False).run()
        self.assertTrue(np.all(td.converged))
        self.assertAlmostEqual(abs(td.e - ref).max(), 0, delta=1e-6)

    def test_m062x_nttda(self):
        mf = self.mol.ROKS(xc='M062X').run()

        ref = np.array([-0.24666086824597583, 0.015820053409613927])
        td = mf.NTTDA().set(nstates=2, deltaS=-1, nobeta=True).run()
        self.assertTrue(np.all(td.converged))
        self.assertAlmostEqual(abs(td.e - ref).max(), 0, delta=1e-6)

        ref = np.array([-0.008184446338165025, 0.025150738879015422])
        td = mf.NTTDA().set(nstates=2, deltaS=0, nobeta=False).run()
        self.assertTrue(np.all(td.converged))
        self.assertAlmostEqual(abs(td.e - ref).max(), 0, delta=1e-6)

        ref = np.array([0.26880002289621757, 0.3280851476633962])
        td = mf.NTTDA().set(nstates=2, deltaS=1, nobeta=False).run()
        self.assertTrue(np.all(td.converged))
        self.assertAlmostEqual(abs(td.e - ref).max(), 0, delta=1e-6)

    def test_cam_b3lyp_nttda(self):
        mf = self.mol.ROKS(xc='CAM-B3LYP').run()

        ref = np.array([-0.0044893465927124268, 0.035037117269294718])
        td = mf.NTTDA().set(nstates=2, deltaS=0, nobeta=True).run()
        self.assertTrue(np.all(td.converged))
        self.assertAlmostEqual(abs(td.e - ref).max(), 0, delta=1e-6)

        ref = np.array([0.27155932081326395, 0.32184531828332463])
        td = mf.NTTDA().set(nstates=2, deltaS=1, nobeta=True).run()
        self.assertTrue(np.all(td.converged))
        self.assertAlmostEqual(abs(td.e - ref).max(), 0, delta=1e-6)

        ref = np.array([-0.22362676199942616, 0.02217598445976246])
        td = mf.NTTDA().set(nstates=2, deltaS=-1, nobeta=False).run()
        self.assertTrue(np.all(td.converged))
        self.assertAlmostEqual(abs(td.e - ref).max(), 0, delta=1e-6)

    def test_hf_nttda_projected_guess(self):
        mf = self.mol.ROKS(xc='HF').run()
        td = mf.NTTDA().set(nstates=2, deltaS=-1, nobeta=False, conv_tol=1e-6)
        vind, hdiag = td.gen_vind_sfd()
        nc = np.count_nonzero(mf.mo_occ == 2)
        no = np.count_nonzero(mf.mo_occ == 1)
        nv = np.count_nonzero(mf.mo_occ == 0)

        # The normalized OO identity is a null vector of the original operator.
        q = np.zeros((nc + no, no + nv))
        q[nc:, :no] = np.eye(no) / np.sqrt(no)
        self.assertAlmostEqual(np.linalg.norm(vind(q.ravel())), 0, delta=1e-12)

        # Deliberately add this redundant direction to every initial guess.
        x0 = td.init_guess(hdiag) + q.ravel()
        original = x0.copy()
        td.kernel(x0=x0)

        ref = np.array([-0.25588162251385815, 0.03179164805915535])
        self.assertTrue(np.all(td.converged))
        self.assertEqual(td.nstates, 2)
        self.assertEqual(len(td.xy), 2)
        self.assertAlmostEqual(abs(td.e - ref).max(), 0, delta=1e-6)
        self.assertEqual(abs(x0 - original).max(), 0)
        for energy, (x, _) in zip(td.e, td.xy):
            self.assertAlmostEqual(np.trace(x[nc:, :no]), 0, delta=1e-12)
            residual = vind(x.ravel()).ravel() - energy * x.ravel()
            self.assertLess(np.linalg.norm(residual), td.conv_tol)

    def test_nttda_check_sanity(self):
        mf = self.mol.ROKS(xc='HF').run()
        td = mf.NTTDA()
        self.assertIs(td.check_sanity(), td)

        td.deltaS = 2
        with self.assertRaisesRegex(ValueError, 'deltaS must be'):
            td.check_sanity()
        with self.assertRaisesRegex(ValueError, 'deltaS must be'):
            td.kernel()
        td.deltaS = -1
        with self.assertRaisesRegex(ValueError, 'nstates must be a positive integer'):
            td.kernel(nstates=0)
        with self.assertRaisesRegex(ValueError, 'nstates must be a positive integer'):
            td.kernel(nstates=1.5)

        mol = gto.M(atom='H 0 0 0', spin=1, basis='sto-3g', verbose=0)
        td = mol.ROKS(xc='HF').run().NTTDA()
        with self.assertRaisesRegex(AssertionError, 'Si>=1'):
            td.kernel()

        mol = gto.M(atom='He 0 0 0', basis='sto-3g', verbose=0)
        td = dft.roks.ROKS(mol, xc='HF').run().NTTDA().set(deltaS=0)
        with self.assertRaisesRegex(AssertionError, 'Si>=1/2'):
            td.kernel()

    def test_nttda_dump_flags(self):
        mf = self.mol.ROKS(xc='HF').run()
        td = mf.NTTDA().set(deltaS=-1, nobeta=True)
        with io.StringIO() as output:
            td.stdout = output
            self.assertIs(td.dump_flags(verbose=logger.INFO), td)
            text = output.getvalue()
            self.assertIn('deltaS = -1 (Si = 1 -> Sf = 0)', text)
            self.assertIn('Numerical stabilization enabled', text)
            self.assertIn('low local beta-electron density', text)
            self.assertIn('Redundant zero-energy state excluded from the excitation calculation', text)
            self.assertNotIn('singlet', text)

            output.seek(0)
            output.truncate()
            td.nobeta = False
            td.verbose = logger.INFO
            td.kernel(nstates=1)
            self.assertTrue(np.all(td.converged))
            self.assertIn('nstates = 1', output.getvalue())
            self.assertNotIn('Numerical stabilization enabled', output.getvalue())

    def test_nttda_physical_zero_root(self):
        # Two open orbitals, no core or virtual orbitals: amplitudes are OO only.
        mol = gto.M(atom='H 0 0 0; H 0 0 1', spin=2, basis='sto-3g', verbose=0)
        mf = mol.ROKS(xc='HF').run()
        td = mf.NTTDA().set(deltaS=-1)

        # Coordinates: (X_00, X_01, X_10, X_11).
        # (1, 0, 0, 1) is the redundant zero mode.
        # The traceless directions have eigenvalues -2, -1, 0, respectively.
        matrix = np.array([
            [-1.,  0., 0.,  1.],
            [ 0., -1., 0.,  0.],
            [ 0.,  0., 0.,  0.],
            [ 1.,  0., 0., -1.],
        ])

        def vind(zs):
            return np.asarray(zs) @ matrix.T

        with patch.object(td, 'gen_vind_sfd', return_value=(vind, matrix.diagonal().copy())):
            # A negative target root must not cause an extra root to be returned.
            td.kernel(nstates=1)
            self.assertTrue(np.all(td.converged))
            self.assertEqual(td.nstates, 1)
            self.assertEqual(len(td.e), 1)
            self.assertAlmostEqual(td.e[0], -2, delta=1e-12)

            # Keep the physical zero root; remove only the OO identity direction.
            ref = np.array([-2., -1., 0.])
            td.kernel(nstates=3)
            self.assertTrue(np.all(td.converged))
            self.assertEqual(td.nstates, 3)
            self.assertAlmostEqual(abs(td.e - ref).max(), 0, delta=1e-12)
            for x, _ in td.xy:
                self.assertAlmostEqual(np.trace(x), 0, delta=1e-12)

            # Four stored coordinates contain only three physical directions.
            td.kernel(nstates=4)
            self.assertTrue(np.all(td.converged))
            self.assertEqual(td.nstates, 3)
            self.assertEqual(len(td.xy), 3)
            self.assertAlmostEqual(abs(td.e - ref).max(), 0, delta=1e-12)


if __name__ == '__main__':
    print('Full tests for noncollinear tensor TDA based on ROKS reference')
    unittest.main()
