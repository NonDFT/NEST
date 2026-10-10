#!/usr/bin/env python
# Copyright 2026 The PySCF Developers. All Rights Reserved.
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
import numpy
from pyscf import gto
from nest.deltascf import sgm


# Q-Chem benchmark, HCHO Rydberg excitation:
#
# $rem
# METHOD BHHLYP
# BASIS cc-pvdz
# SYMMETRY false
# SYM_IGNORE true
# NO_REORIENT True
# XC_GRID 000075000302
# UNRESTRICTED false
# delta_scf true
# $end
#
# $delta_scf
#   triplet restricted
#   triplet_SCF_algorithm SGM
#   somo_1 8
#   somo_2 15
# $end
#
# Q-Chem output:
#   Restricted open-shell triplet state = -113.131417 Ha
#   SOMO(1): best initial orbital 8 overlap 0.977366
#   SOMO(2): best initial orbital 9 overlap 0.991702


class KnownValues(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        atom = '''
        C  0.00000000  0.00000000 -1.13947666
        O  0.00000000  0.00000000  1.14402883
        H  0.00000000  1.76627623 -2.23398653
        H  0.00000000 -1.76627623 -2.23398653
        '''
        cls.mol0 = gto.M(atom=atom, charge=0, spin=0, basis='cc-pvdz',
                         verbose=0, output='/dev/null')
        cls.mf0 = cls.mol0.RKS(xc='BHandHLYP')
        cls.mf0.grids.atom_grid = (75, 302)
        cls.mf0.kernel()

        setocc = cls.mf0.to_uks().mo_occ
        setocc[1][7] -= 1
        setocc[0][14] += 1
        cls.ro_occ = setocc[0] + setocc[1]

        cls.mol1 = gto.M(atom=atom, charge=0, spin=2, basis='cc-pvdz',
                         verbose=0, output='/dev/null')

    @classmethod
    def tearDownClass(cls):
        cls.mol0.stdout.close()
        cls.mol1.stdout.close()
        del cls.mol0, cls.mol1, cls.mf0, cls.ro_occ

    def test_delta_gradient(self):
        mf = self.mol1.ROKS(xc='BHandHLYP')
        mf.grids.atom_grid = (75, 302)
        mf.mo_coeff = self.mf0.mo_coeff.copy()
        mf.mo_occ = self.ro_occ.copy()

        opt = sgm.SGM(mf)
        _h_diag, _delta, grad_delta = opt._exact_sgm_state(mf.mo_coeff,
                                                           mf.mo_occ)
        rng = numpy.random.default_rng(12)
        direction = rng.normal(size=grad_delta.size)
        direction /= numpy.linalg.norm(direction)

        eps = 1e-4
        mo_plus = opt._trial_mo(mf.mo_coeff, mf.mo_occ, eps * direction)
        mo_minus = opt._trial_mo(mf.mo_coeff, mf.mo_occ, -eps * direction)
        g_plus = mf.get_grad(mo_plus, mf.mo_occ)
        g_minus = mf.get_grad(mo_minus, mf.mo_occ)
        fd = (numpy.dot(g_plus, g_plus) - numpy.dot(g_minus, g_minus)) / (2*eps)
        analytic = numpy.dot(grad_delta, direction)
        scale = max(abs(fd), abs(analytic), 1e-12)

        self.assertLess(abs(fd - analytic) / scale, 1e-6,
                        'fd = %.12e, analytic = %.12e' % (fd, analytic))

    def test_hcho_rydberg_triplet(self):
        mf = self.mol1.ROKS(xc='BHandHLYP')
        mf.grids.atom_grid = (75, 302)
        mf.mo_coeff = self.mf0.mo_coeff.copy()
        mf.mo_occ = self.ro_occ.copy()

        opt = sgm.SGM(mf)
        opt.max_cycle = 80
        opt.verbose = 0
        e_tot = opt.kernel()

        self.assertTrue(opt.converged)
        self.assertLess(abs(e_tot - -113.131417), 1e-4)
        self.assertLess(abs(e_tot - -113.131420102713), 1e-6)


if __name__ == '__main__':
    print('Full tests for deltascf.sgm')
    unittest.main()
