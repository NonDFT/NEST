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

    def _check_gradients(self, xc, frozen, expected):
        for i, averaged in enumerate((False, True)):
            mf = self.mol.ROKS(xc=xc)
            if averaged:
                mf = mf.average_occ()
            mf.run()
            self.assertTrue(mf.converged)
            for j, delta_s in enumerate((-1, 0)):
                with self.subTest(xc=xc, averaged=averaged, deltaS=delta_s):
                    td = mf.NTTDA(frozen=frozen).set(deltaS=delta_s).run()
                    self.assertTrue(np.all(td.converged))
                    analytic = td.Gradients().kernel(state=2)
                    np.testing.assert_allclose(analytic, expected[i, j], atol=1e-5, rtol=0)

    def test_core_gradients(self):
        # Analytic-gradient regression with default SCF/NTTDA tolerances.
        expected = np.array([
            [[[0, .1349893, -.07367758], [0, .04537355, -.03275083], [0, -.1803629, .1064284]],
             [[0, .3304962, .1163615], [0, -.06797934, -.1135967], [0, -.2625168, -.002764826]]],
            [[[0, .1362866, -.07205570], [0, .04493596, -.03411088], [0, -.1812226, .1061666]],
             [[0, .3349085, .1250914], [0, -.06995655, -.1207165], [0, -.2649519, -.004374936]]],
        ])
        self._check_gradients('HF', frozen=1, expected=expected)

    def test_frozen_occupied_and_virtual_gradients(self):
        # Freeze one valence occupied orbital and the last virtual orbital.
        expected = np.array([
            [[[0, .1506957, -.05324929], [0, .03755716, -.04627062], [0, -.1882710, .09954212]],
             [[0, .3361471, .1288448], [0, -.06850790, -.1276500], [0, -.2672964, -.001574045]]],
            [[[0, .1521700, -.05221959], [0, .03720607, -.04724038], [0, -.1893800, .09946592]],
             [[0, .3344835, .1267725], [0, -.06867178, -.1241494], [0, -.2658151, -.002614508]]],
        ])
        self._check_gradients('PBE', frozen=[1, -1], expected=expected)


if __name__ == '__main__':
    unittest.main()
