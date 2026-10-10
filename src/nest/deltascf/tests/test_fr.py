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
from pyscf import gto
from nest.deltascf import fr


class KnownValues(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        atom = '''
        O  0.000  0.000  0.000
        H  0.000 -0.757  0.587
        H  0.000  0.757  0.587
        '''
        cls.mol0 = gto.M(atom=atom, basis='6-31g', spin=0, verbose=0)
        cls.mol1 = gto.M(atom=atom, basis='6-31g', spin=2, verbose=0)

    def test_hcho_rydberg_triplet(self):
        # Same Rydberg excitation as test_sgm.py; Q-Chem SOMOs 8 and 15.
        # Oracle: vsQChem/HCHO/QChem/hcho.log, triplet energy -113.131417 Ha.
        atom = '''
        C  0.00000000  0.00000000 -1.13947666
        O  0.00000000  0.00000000  1.14402883
        H  0.00000000  1.76627623 -2.23398653
        H  0.00000000 -1.76627623 -2.23398653
        '''
        mol = gto.M(atom=atom, basis='cc-pvdz', spin=0, verbose=0)
        ground = mol.RKS(xc='BHandHLYP')
        ground.grids.atom_grid = (75, 302)
        ground.kernel()
        occupations = ground.mo_occ.copy()
        occupations[7] = 1
        occupations[14] = 1

        triplet = mol.copy()
        triplet.spin = 2
        mf = triplet.ROKS(xc='BHandHLYP')
        mf.grids.atom_grid = (75, 302)
        opt = fr.FR(mf)
        opt.max_cycle = 100
        e_tot = opt.kernel(ground.mo_coeff, occupations)

        self.assertTrue(opt.freeze_converged)
        self.assertTrue(opt.converged)
        self.assertAlmostEqual(e_tot, -113.131417, delta=1e-4)
        # PySCF SGM reference for the same state and numerical settings.
        self.assertAlmostEqual(e_tot, -113.131420102713, delta=1e-6)

    def test_h2o_triplet_rohf(self):
        ground = self.mol0.RHF().run()
        occupations = ground.mo_occ.copy()
        occupations[4] = 1  # HOMO -> LUMO: 2/0 becomes 1/1.
        occupations[5] = 1

        opt = fr.FR(self.mol1.ROHF())
        opt.conv_tol = 1e-10
        opt.conv_tol_grad = 1e-6
        opt.max_cycle = 100
        e_tot = opt.kernel(ground.mo_coeff, occupations)

        # This valence excitation reaches the ordinary triplet SCF solution.
        reference = self.mol1.ROHF()
        reference.conv_tol = 1e-11
        reference.conv_tol_grad = 1e-7
        reference.kernel()

        self.assertTrue(opt.freeze_converged)
        self.assertTrue(opt.converged)
        self.assertTrue(reference.converged)
        self.assertAlmostEqual(e_tot, reference.e_tot, places=7)

    def test_h2o_triplet_roks(self):
        ground = self.mol0.RKS(xc='BHandHLYP')
        ground.grids.level = 0
        ground.kernel()
        occupations = ground.mo_occ.copy()
        occupations[4] = 1  # HOMO -> LUMO: 2/0 becomes 1/1.
        occupations[5] = 1

        mf = self.mol1.ROKS(xc='BHandHLYP')
        mf.grids.level = 0
        opt = fr.FR(mf)
        opt.conv_tol = 1e-10
        opt.conv_tol_grad = 1e-6
        opt.max_cycle = 100
        e_tot = opt.kernel(ground.mo_coeff, occupations)

        reference = self.mol1.ROKS(xc='BHandHLYP')
        reference.grids.level = 0
        reference.conv_tol = 1e-11
        reference.conv_tol_grad = 1e-7
        reference.kernel()

        self.assertTrue(opt.freeze_converged)
        self.assertTrue(opt.converged)
        self.assertTrue(reference.converged)
        self.assertAlmostEqual(e_tot, reference.e_tot, places=7)


if __name__ == '__main__':
    print('Full tests for deltascf.fr')
    unittest.main()
