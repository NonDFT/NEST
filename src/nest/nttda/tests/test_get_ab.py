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

    def test_nttda_get_ab_df(self):
        for xc in ('HF', 'PBE', 'CAM-B3LYP'):
            mf = self.mol.ROKS(xc=xc).density_fit(auxbasis='weigend').run(conv_tol=1e-11)
            self.assertTrue(mf.converged)
            td = mf.NTTDA()
            for nobeta in (False, True):
                td.nobeta = nobeta
                for delta_s, generator in ((-1, td.gen_vind_sfd), (0, td.gen_vind_sc), (1, td.gen_vind_sfu)):
                    td.deltaS = delta_s
                    vind, hdiag = generator()
                    a = td.get_ab()
                    self.assertAlmostEqual(abs(a - a.T).max(), 0, delta=1e-11)
                    x = np.random.default_rng(12).standard_normal((10, hdiag.size))
                    self.assertAlmostEqual(abs(vind(x) - (a @ x.T).T).max(), 0, delta=1e-11)

        # Mixing fitted J with exact K cannot be represented by one K^Ref tensor.
        mf.only_dfj = True
        with self.assertRaises(NotImplementedError):
            td.get_ab()

    def test_nttda_get_ab(self):
        for xc in ('HF', 'SVWN', 'PBE', 'M062X', 'CAM-B3LYP'):
            mf = self.mol.ROKS(xc=xc).run(conv_tol=1e-11)
            self.assertTrue(mf.converged)
            for nobeta in (False, True):
                for delta_s in (-1, 0, 1):
                    td = mf.NTTDA().set(deltaS=delta_s, nobeta=nobeta)
                    if delta_s == -1:
                        vind, hdiag = td.gen_vind_sfd()
                    elif delta_s == 0:
                        vind, hdiag = td.gen_vind_sc()
                    else:
                        vind, hdiag = td.gen_vind_sfu()

                    a = td.get_ab()
                    self.assertEqual(a.shape, (hdiag.size, hdiag.size))
                    self.assertAlmostEqual(abs(a - a.T).max(), 0, delta=1e-11)
                    x = np.random.default_rng(12).standard_normal((10, hdiag.size))
                    self.assertAlmostEqual(abs(vind(x) - (a @ x.T).T).max(), 0, delta=1e-11)

                    if delta_s == -1:
                        nc = np.count_nonzero(mf.mo_occ == 2)
                        no = np.count_nonzero(mf.mo_occ == 1)
                        nv = np.count_nonzero(mf.mo_occ == 0)
                        q = np.zeros((nc + no, no + nv))
                        q[nc:, :no] = np.eye(no) / np.sqrt(no)
                        self.assertAlmostEqual(np.linalg.norm(a @ q.ravel()), 0, delta=1e-11)


if __name__ == '__main__':
    unittest.main()
