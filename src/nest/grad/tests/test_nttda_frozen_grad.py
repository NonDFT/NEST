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
from nest import aocscf, nttda


class FrozenNTTDAGradients(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Same triplet CH2 geometry as test_nttda_aocscf_fd.py.
        cls.mol = gto.M(
            atom='C 0 0 0; H 0 0 2.0; H 0 1.7 -0.5',
            basis='sto-3g',
            spin=2,
            unit='Bohr',
            verbose=0,
        )

    def _check_gradients(self, xc, frozen):
        for averaged in (False, True):
            mf = self.mol.ROKS(xc=xc)
            if averaged:
                mf = mf.average_occ()
            # Finite differences amplify SCF and excitation-energy errors.
            mf.run(conv_tol=1e-13, conv_tol_grad=1e-10)
            self.assertTrue(mf.converged)
            for delta_s in (-1, 0):
                with self.subTest(xc=xc, averaged=averaged, deltaS=delta_s):
                    td = mf.NTTDA(frozen=frozen).set(
                        deltaS=delta_s, conv_tol=1e-8, lindep=1e-20,
                    ).run()
                    self.assertTrue(np.all(td.converged))
                    gradient = td.Gradients()
                    analytic = gradient.kernel(state=2)
                    finite_difference = gradient.kernel(state=2, method='finite_diff', step=2e-4)
                    np.testing.assert_allclose(analytic, finite_difference, atol=5e-6, rtol=0)

    def test_core_gradients(self):
        self._check_gradients('HF', frozen=1)

    def test_frozen_occupied_and_virtual_gradients(self):
        # Freeze one valence occupied orbital and the last virtual orbital.
        self._check_gradients('PBE', frozen=[1, -1])


if __name__ == '__main__':
    unittest.main()
