# Copyright 2021-2024 The PySCF Developers. All Rights Reserved.
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

def diagonalize_tddft(mf, extype=1, collinear="mcol", collinear_samples=50, nstates=5):
    a, b = sftda.uhf_sf.get_ab_sf(mf, collinear=collinear, collinear_samples=collinear_samples)
    A_baba, A_abab = a
    B_baab, B_abba = b
    n_occ_a, n_virt_b = A_abab.shape[0], A_abab.shape[1]
    n_occ_b, n_virt_a = B_abba.shape[2], B_abba.shape[3]
    A_abab_2d = A_abab.reshape((n_occ_a*n_virt_b, n_occ_a*n_virt_b), order='C')
    B_abba_2d = B_abba.reshape((n_occ_a*n_virt_b, n_occ_b*n_virt_a), order='C')
    B_baab_2d = B_baab.reshape((n_occ_b*n_virt_a, n_occ_a*n_virt_b), order='C')
    A_baba_2d = A_baba.reshape((n_occ_b*n_virt_a, n_occ_b*n_virt_a), order='C')
    Casida_matrix = np.block([[ A_abab_2d, B_abba_2d],
                              [-B_baab_2d, -A_baba_2d]])
    eigenvals, eigenvecs = np.linalg.eig(Casida_matrix)
    idx = eigenvals.real.argsort()
    eigenvals = eigenvals[idx]
    eigenvecs = eigenvecs[:, idx]
    norms = np.linalg.norm(eigenvecs[:n_occ_a*n_virt_b], axis=0)**2
    norms -= np.linalg.norm(eigenvecs[n_occ_a*n_virt_b:], axis=0)**2
    if extype == 1:
        mask = norms > 1e-3
        valid_e = eigenvals[mask].real
    else:
        mask = norms < -1e-3
        valid_e = eigenvals[mask].real
        valid_e = -valid_e
    lowest_e = np.sort(valid_e)[:nstates]
    return lowest_e

class KnownValues(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        mol = gto.Mole()
        mol.verbose = 0
        mol.output = '/dev/null'
        mol.atom = '''
        O     0.   0.       0.
        H     0.   -0.757   0.587
        H     0.   0.757    0.587'''
        mol.spin = 2
        mol.basis = '631g'
        cls.mol = mol.build()

    @classmethod
    def tearDownClass(cls):
        cls.mol.stdout.close()

    def test_hf_tddft(self):
        mf = self.mol.UKS(xc='HF').run()
        ref = np.array([-0.2170069142,  0.0000000351])
        td = mf.TDDFT_SF().set(extype=1, collinear_samples=50, nstates=2, conv_tol=1e-6).run()
        self.assertTrue(np.all(td.converged))
        self.assertAlmostEqual(abs(td.e - ref).max(), 0, delta=1e-6)
        e = diagonalize_tddft(mf, extype=1, collinear_samples=50, nstates=2)
        self.assertAlmostEqual(abs(e - td.e).max(), 0, delta=1e-6)

    def test_mcol_lda_tddft(self):
        mf = self.mol.UKS(xc='SVWN').run()
        ref = np.array([0.4496080757, 0.5767661981])
        td = mf.TDDFT_SF().set(extype=0, collinear_samples=50, nstates=2, conv_tol=1e-6).run()
        self.assertTrue(np.all(td.converged))
        self.assertAlmostEqual(abs(td.e - ref).max(), 0, delta=1e-6)
        e = diagonalize_tddft(mf, extype=0, collinear_samples=50, nstates=2)
        self.assertAlmostEqual(abs(e - td.e).max(), 0, delta=1e-6)

    def test_mcol_b3lyp_tddft(self):
        mf = self.mol.UKS(xc='B3LYP').run()
        ref = np.array([-0.2966232093, -0.0000000014])
        td = mf.TDDFT_SF().set(extype=1, collinear_samples=50, nstates=2, conv_tol=1e-6).run()
        self.assertTrue(np.all(td.converged))
        self.assertAlmostEqual(abs(td.e - ref).max(), 0, delta=1e-6)
        e = diagonalize_tddft(mf, extype=1, collinear_samples=50, nstates=2)
        self.assertAlmostEqual(abs(e - td.e).max(), 0, delta=1e-6)

    def test_col_b3lyp_tddft(self):
        mf = self.mol.UKS(xc='B3LYP').run()
        ref = np.array([0.4733593072, 0.6059910743])
        td = mf.TDDFT_SF().set(extype=0, collinear="col", collinear_samples=50, nstates=2, conv_tol=1e-6).run()
        self.assertTrue(np.all(td.converged))
        self.assertAlmostEqual(abs(td.e - ref).max(), 0, delta=1e-6)
        e = diagonalize_tddft(mf, extype=0, collinear="col", collinear_samples=50, nstates=2)
        self.assertAlmostEqual(abs(e - td.e).max(), 0, delta=1e-6)

    def test_mcol_tpss_tddft(self):
        mf = self.mol.UKS(xc='TPSS').run()
        ref = np.array([-0.2873950303,  0.0000000036])
        td = mf.TDDFT_SF().set(extype=1, collinear_samples=50, nstates=2, conv_tol=1e-6).run()
        self.assertTrue(np.all(td.converged))
        self.assertAlmostEqual(abs(td.e - ref).max(), 0, delta=1e-6)
        e = diagonalize_tddft(mf, extype=1, collinear_samples=50, nstates=2)
        self.assertAlmostEqual(abs(e - td.e).max(), 0, delta=1e-6)

    def test_mcol_cam_tddft(self):
        mf = self.mol.UKS(xc='CAM-B3LYP').run()
        ref = np.array([0.4595188942, 0.5715321744])
        td = mf.TDDFT_SF().set(extype=0, collinear_samples=50, nstates=2, conv_tol=1e-6).run()
        self.assertTrue(np.all(td.converged))
        self.assertAlmostEqual(abs(td.e - ref).max(), 0, delta=1e-6)
        e = diagonalize_tddft(mf, extype=0, collinear_samples=50, nstates=2)
        self.assertAlmostEqual(abs(e - td.e).max(), 0, delta=1e-6)

    def test_col_cam_tddft(self):
        mf = self.mol.UKS(xc='CAM-B3LYP').run()
        ref = np.array([0.4749112891, 0.6040290720])
        td = mf.TDDFT_SF().set(extype=0, collinear="col", collinear_samples=50, nstates=2, conv_tol=1e-6).run()
        self.assertTrue(np.all(td.converged))
        self.assertAlmostEqual(abs(td.e - ref).max(), 0, delta=1e-6)
        e = diagonalize_tddft(mf, extype=0, collinear="col", collinear_samples=50, nstates=2)
        self.assertAlmostEqual(abs(e - td.e).max(), 0, delta=1e-6)

    def test_hf_tddft_roks(self):
        mf = self.mol.ROKS(xc='HF').run()
        ref = np.array([0.4629613282, 0.5364066167])
        td = sftda.TDDFT_SF(mf).set(extype=0, collinear_samples=50, nstates=2, conv_tol=1e-6).run()
        self.assertTrue(np.all(td.converged))
        self.assertAlmostEqual(abs(td.e - ref).max(), 0, delta=1e-6)
        e = diagonalize_tddft(mf, extype=0, collinear_samples=50, nstates=2)
        self.assertAlmostEqual(abs(e - td.e).max(), 0, delta=1e-6)

    def test_mcol_lda_tddft_roks(self):
        mf = self.mol.ROKS(xc='SVWN').run()
        ref = np.array([-0.3273393340, -0.0007543304])
        td = sftda.TDDFT_SF(mf).set(extype=1, collinear_samples=50, nstates=2, conv_tol=1e-6).run()
        self.assertTrue(np.all(td.converged))
        self.assertAlmostEqual(abs(td.e - ref).max(), 0, delta=1e-6)
        e = diagonalize_tddft(mf, extype=1, collinear_samples=50, nstates=2)
        self.assertAlmostEqual(abs(e - td.e).max(), 0, delta=1e-6)

    def test_mcol_b3lyp_tddft_roks(self):
        mf = self.mol.ROKS(xc='B3LYP').run()
        ref = np.array([0.4587747539, 0.5730180241])
        td = sftda.TDDFT_SF(mf).set(extype=0, collinear_samples=50, nstates=2, conv_tol=1e-6).run()
        self.assertTrue(np.all(td.converged))
        self.assertAlmostEqual(abs(td.e - ref).max(), 0, delta=1e-6)
        e = diagonalize_tddft(mf, extype=0, collinear_samples=50, nstates=2)
        self.assertAlmostEqual(abs(e - td.e).max(), 0, delta=1e-6)

    def test_col_b3lyp_tddft_roks(self):
        mf = self.mol.ROKS(xc='B3LYP').run()
        ref = np.array([-0.2865622699,  0.0415443030])
        td = sftda.TDDFT_SF(mf).set(extype=1, collinear="col", collinear_samples=50, nstates=2, conv_tol=1e-6).run()
        self.assertTrue(np.all(td.converged))
        self.assertAlmostEqual(abs(td.e - ref).max(), 0, delta=1e-6)
        e = diagonalize_tddft(mf, extype=1, collinear="col", collinear_samples=50, nstates=2)
        self.assertAlmostEqual(abs(e - td.e).max(), 0, delta=1e-6)

    def test_mcol_tpss_tddft_roks(self):
        mf = self.mol.ROKS(xc='TPSS').run()
        ref = np.array([0.4486464970, 0.5651121496])
        td = sftda.TDDFT_SF(mf).set(extype=0, collinear_samples=50, nstates=2, conv_tol=1e-6).run()
        self.assertTrue(np.all(td.converged))
        self.assertAlmostEqual(abs(td.e - ref).max(), 0, delta=1e-6)
        e = diagonalize_tddft(mf, extype=0, collinear_samples=50, nstates=2)
        self.assertAlmostEqual(abs(e - td.e).max(), 0, delta=1e-6)

    def test_mcol_cam_tddft_roks(self):
        mf = self.mol.ROKS(xc='CAM-B3LYP').run()
        ref = np.array([-0.2992951728, -0.0013654305])
        td = sftda.TDDFT_SF(mf).set(extype=1, collinear_samples=50, nstates=2, conv_tol=1e-6).run()
        self.assertTrue(np.all(td.converged))
        self.assertAlmostEqual(abs(td.e - ref).max(), 0, delta=1e-6)
        e = diagonalize_tddft(mf, extype=1, collinear_samples=50, nstates=2)
        self.assertAlmostEqual(abs(e - td.e).max(), 0, delta=1e-6)

    def test_col_cam_tddft_roks(self):
        mf = self.mol.ROKS(xc='CAM-B3LYP').run()
        ref = np.array([-0.2888221413,  0.0288018909])
        td = sftda.TDDFT_SF(mf).set(extype=1, collinear="col", collinear_samples=50, nstates=2, conv_tol=1e-6).run()
        self.assertTrue(np.all(td.converged))
        self.assertAlmostEqual(abs(td.e - ref).max(), 0, delta=1e-6)
        e = diagonalize_tddft(mf, extype=1, collinear="col", collinear_samples=50, nstates=2)
        self.assertAlmostEqual(abs(e - td.e).max(), 0, delta=1e-6)

    def test_tddft_scanner(self):
        mf = self.mol.UKS(xc='HF').run()
        for extype in (0, 1):
            td = mf.SFTDDFT().set(extype=extype, nstates=3)
            ref = td.kernel()[0].copy()
            td_scan = td.as_scanner()
            td_scan.max_cycle = 1
            td_scan(self.mol)
            self.assertAlmostEqual(abs(td_scan.e - ref).max(), 0, delta=1e-6)

if __name__ == "__main__":
    print("Full Tests for spin-flip-TDDFT with UKS and ROKS references")
    unittest.main()
