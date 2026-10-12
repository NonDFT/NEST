# Copyright 2026 The NEST Developers. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest
import numpy as np
from pyscf import gto
from nest.deltascf import fr, sgm


class Sfx2cDeltaScfTests(unittest.TestCase):
    def test_fr_and_sgm_preserve_sfx2c_hamiltonian(self):
        # This valence triplet can also be reached by ordinary ROKS, giving
        # an independent SCF reference without relying on optimizer internals.
        mol = gto.M(atom='O 0 0 0; H 0 -.757 .587; H 0 .757 .587', basis='6-31g', verbose=0)
        ground = mol.RKS(xc='BHandHLYP').sfx2c1e()
        ground.chkfile = None
        ground.grids.level = 0
        ground.conv_tol = 1e-11
        ground.kernel()
        self.assertTrue(ground.converged)
        occ = ground.mo_occ.copy()
        occ[4] = occ[5] = 1
        triplet = mol.copy()
        triplet.spin = 2
        reference = triplet.ROKS(xc=ground.xc).sfx2c1e()
        reference.chkfile = None
        reference.grids.level = 0
        reference.conv_tol = 1e-11
        reference.conv_tol_grad = 1e-7
        reference.kernel()
        self.assertTrue(reference.converged)
        hcore = reference.get_hcore()
        self.assertGreater(np.max(abs(hcore - reference.undo_x2c().get_hcore())), 1e-4)
        for optimizer in (fr.FR, sgm.SGM):
            with self.subTest(optimizer=optimizer.__name__):
                mf = triplet.ROKS(xc=ground.xc).sfx2c1e()
                mf.chkfile = None
                mf.grids.level = 0
                opt = optimizer(mf)
                opt.conv_tol = 1e-10
                opt.conv_tol_grad = 1e-6
                opt.kernel(ground.mo_coeff, occ)
                self.assertTrue(opt.converged)
                if optimizer is fr.FR:
                    self.assertTrue(opt.freeze_converged)
                    final = opt
                else:
                    final = opt.undo_sgm()
                self.assertIs(final.with_x2c, mf.with_x2c)
                np.testing.assert_allclose(final.get_hcore(), hcore, atol=1e-13, rtol=0)
                self.assertAlmostEqual(final.e_tot, reference.e_tot, delta=1e-7)
                self.assertLess(np.linalg.norm(final.get_grad(final.mo_coeff, final.mo_occ)), 1e-6)


if __name__ == '__main__':
    unittest.main()
