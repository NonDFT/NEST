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


HF_REF_ETF = np.array([
    [-3.95216073e-03, -3.75136071e-04, -9.45804349e-06],
    [ 2.29982740e-03,  1.83601713e-04,  2.18434800e-06],
    [-3.38828168e-04,  2.19688758e-02, -7.70725017e-04],
    [ 1.99116150e-03, -2.17773415e-02,  7.77998712e-04],
])

HF_REF_FULL = np.array([
    [ 4.07870800e-03,  3.87408301e-04,  8.03498796e-06],
    [-2.12054721e-03, -1.74366077e-04, -2.81871440e-06],
    [ 5.19504915e-04, -2.61168634e-02,  6.30604272e-04],
    [-2.25106258e-03,  2.59162799e-02, -6.37937351e-04],
])

B3LYP_REF_ETF = np.array([
    [ 2.50608626e-03,  3.33283037e-04,  3.17791471e-05],
    [-9.06831663e-04, -1.32591849e-04, -1.33038220e-05],
    [ 1.11121960e-03, -3.60580919e-02, -3.53482487e-04],
    [-2.71053372e-03,  3.58574323e-02,  3.35164345e-04],
])

B3LYP_REF_FULL = np.array([
    [-2.98860503e-03, -3.83780059e-04, -2.82220009e-05],
    [ 7.40415810e-04,  1.27679743e-04,  1.47507459e-05],
    [-1.53440277e-03,  4.54561852e-02,  6.15100183e-04],
    [ 3.28538114e-03, -4.52414485e-02, -5.96649158e-04],
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
        mf.kernel()
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
        self.assertAlmostEqual(diff, 0, delta=1e-8)

    def test_hf_full(self):
        _, diff = self._check_nac('HF', False, HF_REF_FULL)
        self.assertAlmostEqual(diff, 0, delta=1e-8)

    def test_b3lyp_etf(self):
        _, diff = self._check_nac('B3LYP', True, B3LYP_REF_ETF)
        self.assertAlmostEqual(diff, 0, delta=1e-8)

    def test_b3lyp_full(self):
        _, diff = self._check_nac('B3LYP', False, B3LYP_REF_FULL)
        self.assertAlmostEqual(diff, 0, delta=1e-8)

    def test_extype0_smoke(self):
        mol = gto.M(
            atom='O 0 0 0; H 0 -0.757 0.587; H 0 0.757 0.587',
            basis='sto-3g',
            spin=2,
            verbose=0,
            output='/dev/null',
        )
        mf = dft.UKS(mol).set(xc='HF').run()
        td = mf.TDDFT_SF().set(
            extype=0, collinear='mcol', collinear_samples=20, nstates=3
        ).run()
        value = td.NAC().kernel(state_I=1, state_J=2, ediff=False, use_etfs=False)
        self.assertEqual(value.shape, (mol.natm, 3))
        self.assertTrue(np.isfinite(value).all())

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
