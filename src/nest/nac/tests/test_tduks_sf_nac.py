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

from pyscf import dft
from pyscf import gto

from nest import sftda
from nest.nac import tduks_sf as packaged_nac
from nest.sftda.uhf_sf import get_ab_sf


# PySCF-forge #231 references, PySCF 2.15.0 with SCF conv_tol=1e-12.
HF_REF_ETF = np.array([
    [-3.95470206e-03, -3.73615571e-04, -9.69878457e-06],
    [ 2.30042868e-03,  1.82489975e-04,  3.11897170e-06],
    [-3.38391814e-04,  2.19689308e-02, -7.70962711e-04],
    [ 1.99266520e-03, -2.17778053e-02,  7.77542524e-04],
])

HF_REF_FULL = np.array([
    [ 4.08124376e-03,  3.85893263e-04,  8.26819990e-06],
    [-2.12115474e-03, -1.73151723e-04, -3.75968091e-06],
    [ 5.19067725e-04, -2.61169259e-02,  6.30842571e-04],
    [-2.25256655e-03,  2.59167244e-02, -6.37481239e-04],
])

B3LYP_REF_ETF = np.array([
    [ 2.50598126e-03,  3.33252101e-04,  3.17780136e-05],
    [-9.06812181e-04, -1.32553941e-04, -1.33029415e-05],
    [ 1.11124891e-03, -3.60578572e-02, -3.53475871e-04],
    [-2.71047750e-03,  3.58571905e-02,  3.35157983e-04],
])

B3LYP_REF_FULL = np.array([
    [-2.98849214e-03, -3.83745104e-04, -2.82205655e-05],
    [ 7.40404358e-04,  1.27644164e-04,  1.47500840e-05],
    [-1.53443210e-03,  4.54559163e-02,  6.15094567e-04],
    [ 3.28532155e-03, -4.52411741e-02, -5.96643805e-04],
])


def solve_shared_tddft(mf, extype=1, collinear_samples=50):
    a, b = get_ab_sf(mf, collinear_samples=collinear_samples)
    A_baba, A_abab = a
    B_baab, B_abba = b

    mo_occ = mf.mo_occ
    n_occ_a = int((mo_occ[0] > 0).sum())
    n_virt_a = int((mo_occ[0] == 0).sum())
    n_occ_b = int((mo_occ[1] > 0).sum())
    n_virt_b = int((mo_occ[1] == 0).sum())

    A_abab_2d = A_abab.reshape((n_occ_a * n_virt_b, n_occ_a * n_virt_b))
    B_abba_2d = B_abba.reshape((n_occ_a * n_virt_b, n_occ_b * n_virt_a))
    B_baab_2d = B_baab.reshape((n_occ_b * n_virt_a, n_occ_a * n_virt_b))
    A_baba_2d = A_baba.reshape((n_occ_b * n_virt_a, n_occ_b * n_virt_a))

    casida_matrix = np.block([
        [A_abab_2d, B_abba_2d],
        [-B_baab_2d, -A_baba_2d],
    ])
    eigenvals, eigenvecs = np.linalg.eig(casida_matrix)
    idx = eigenvals.real.argsort()
    eigenvals = eigenvals[idx].real
    eigenvecs = eigenvecs[:, idx]

    norms = np.linalg.norm(eigenvecs[:n_occ_a * n_virt_b], axis=0) ** 2
    norms -= np.linalg.norm(eigenvecs[n_occ_a * n_virt_b:], axis=0) ** 2
    valid_mask = norms > 0 if extype == 1 else norms < 0

    return (
        eigenvals[valid_mask],
        eigenvecs[:, valid_mask].T,
        n_occ_a,
        n_virt_a,
        n_occ_b,
        n_virt_b,
    )


def build_td_object(mf, solved_data, extype=1, collinear_samples=50, xy_format='new'):
    e, vecs, n_occ_a, n_virt_a, n_occ_b, n_virt_b = solved_data

    def norm_xy_old(z):
        x_flat = z[:n_occ_a * n_virt_b]
        y_flat = z[n_occ_a * n_virt_b:]
        norm_val = np.linalg.norm(x_flat) ** 2 - np.linalg.norm(y_flat) ** 2
        norm_val = np.sqrt(1.0 / norm_val)
        x = x_flat.reshape(n_occ_a, n_virt_b) * norm_val
        y = y_flat.reshape(n_occ_b, n_virt_a) * norm_val
        return ((0, x), (y, 0)) if extype == 1 else ((y, 0), (0, x))

    def norm_xy_new(z):
        x_flat = z[:n_occ_a * n_virt_b]
        y_flat = z[n_occ_a * n_virt_b:]
        norm_val = np.linalg.norm(x_flat) ** 2 - np.linalg.norm(y_flat) ** 2
        norm_val = np.sqrt(1.0 / norm_val)
        x = x_flat.reshape(n_occ_a, n_virt_b) * norm_val
        y = y_flat.reshape(n_occ_b, n_virt_a) * norm_val
        return (x, y) if extype == 1 else (y, x)

    td = sftda.uhf_sf.TDDFT_SF(mf)
    td.e = e
    td.xy = [norm_xy_old(z) if xy_format == 'old' else norm_xy_new(z) for z in vecs]
    td.nstates = len(e)
    td.extype = extype
    td.collinear_samples = collinear_samples
    return td


class KnownValues(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        mol = gto.Mole()
        mol.atom = '''
C      0.000000    0.000000    0.000000
O      0.000000    0.000000    1.205000
H     -0.937704    0.000000   -0.513544
H      0.937704    0.100000   -0.513544
'''
        mol.basis = 'cc-pvdz'
        mol.spin = 2
        mol.verbose = 0
        mol.output = '/dev/null'
        cls.mol = mol.build()

    @classmethod
    def tearDownClass(cls):
        cls.mol.stdout.close()

    def _check_nac(self, xc, use_etfs, ref):
        mf = dft.UKS(self.mol)
        mf.xc = xc
        mf.conv_tol = 1e-12
        mf.kernel()
        self.assertTrue(mf.converged)
        solved_data = solve_shared_tddft(mf, extype=1, collinear_samples=50)
        td_new = build_td_object(mf, solved_data, extype=1, collinear_samples=50, xy_format='new')
        new_val = packaged_nac.NAC(td_new).kernel(
            state_I=1, state_J=3, use_etfs=use_etfs, ediff=False
        )
        diff_direct = np.max(np.abs(new_val - ref))
        diff_flipped = np.max(np.abs(-new_val - ref))
        if diff_flipped < diff_direct:
            new_val = -new_val
            diff = diff_flipped
        else:
            diff = diff_direct
        return new_val, diff

    def test_hf_etf(self):
        _, diff = self._check_nac('HF', True, HF_REF_ETF)
        self.assertAlmostEqual(diff, 0, delta=1e-7)

    def test_hf_full(self):
        _, diff = self._check_nac('HF', False, HF_REF_FULL)
        self.assertAlmostEqual(diff, 0, delta=1e-7)

    def test_b3lyp_etf(self):
        _, diff = self._check_nac('B3LYP', True, B3LYP_REF_ETF)
        self.assertAlmostEqual(diff, 0, delta=1e-7)

    def test_b3lyp_full(self):
        _, diff = self._check_nac('B3LYP', False, B3LYP_REF_FULL)
        self.assertAlmostEqual(diff, 0, delta=1e-7)

    def test_mcol_lda_tda(self):
        mol = gto.M(atom='O 0 0 0; H 0 -0.757 0.587; H 0 0.757 0.587',
                    basis='6-31g', spin=2, verbose=0)
        mf = mol.UKS(xc='SVWN').run()
        td = mf.SFTDA().set(nstates=3).run()
        self.assertTrue(mf.converged and all(td.converged))
        de = td.NAC().kernel(state_I=1, state_J=3)
        # Full NAC numerator from gpu4pyscf's SF-TDA water benchmark.
        ref_de = np.array([[-0.016524445, 0., 0.], [0.0314787174, 0., 0.], [0.0314787174, 0., 0.]])
        self.assertAlmostEqual(abs(abs(de) - abs(ref_de)).max(), 0, delta=1e-5)

    def test_extype1_call_flow(self):
        mol = gto.Mole()
        mol.atom = '''
O     0.000000    0.000000    0.000000
H     0.000000   -0.757000    0.587000
H     0.000000    0.757000    0.587000
'''
        mol.basis = '631g'
        mol.spin = 2
        mol.verbose = 0
        mol.output = '/dev/null'
        mol.build()

        mf = dft.UKS(mol)
        mf.xc = 'B3LYP'
        mf.kernel()

        td = mf.TDDFT_SF().set(
            extype=1, collinear_samples=50, nstates=3
        ).run()
        nac = packaged_nac.NAC(td).kernel(
            state_I=1, state_J=2, use_etfs=False, ediff=False
        )
        self.assertEqual(nac.shape, (mol.natm, 3))
        self.assertTrue(np.isfinite(nac).all())


if __name__ == '__main__':
    print('Full tests for SF-TDDFT nonadiabatic couplings')
    unittest.main()
