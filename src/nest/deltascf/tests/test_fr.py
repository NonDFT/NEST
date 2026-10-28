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
        e_tot = opt.kernel(ground.mo_coeff, occupations)

        self.assertTrue(opt.freeze_converged)
        self.assertTrue(opt.converged)
        self.assertAlmostEqual(e_tot, -113.131417, delta=1e-4)
        # PySCF SGM reference for the same state and numerical settings.
        self.assertAlmostEqual(e_tot, -113.131420102713, delta=1e-6)

    def test_hcho_rydberg_triplet_aocscf(self):
        # Same target state as test_hcho_rydberg_triplet, with average-occupation orbitals.
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
        mf = triplet.ROKS(xc='BHandHLYP').average_occ()
        mf.grids.atom_grid = (75, 302)
        opt = mf.FR()
        opt.kernel(ground.mo_coeff, occupations)

        self.assertTrue(opt.freeze_converged)
        self.assertTrue(opt.converged)
        self.assertAlmostEqual(opt.e_avg_occ, -113.03118394153302, delta=1e-6)
        # The high-spin energy is not stationary at the AOCSCF orbitals.
        self.assertAlmostEqual(opt.e_tot, -113.12691855942077, delta=1e-4)


if __name__ == '__main__':
    print('Full tests for deltascf.fr')
    unittest.main()
