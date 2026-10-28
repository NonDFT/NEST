# Copyright 2021-2024 The PySCF Developers. All Rights Reserved.
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
from nest import aocscf
from nest.nttda import nttda


# Lowest three excitation energies (Hartree) for the frozen selections below.
REFS = {
    'HF': {
        -1: np.array([-0.24677483412090, 0.04144784701196, 0.09557456822705]),
        0: np.array([-0.01550655645228, 0.03867464949256, 0.06748186704164]),
        1: np.array([0.26473058502390, 0.32221477286873, 0.35874323960772]),
    },
    'SVWN': {
        -1: np.array([-0.21189754289401, 0.02406315531387, 0.06980001135606]),
        0: np.array([-0.00128684220396, 0.04223185240056, 0.08782576896338]),
        1: np.array([0.31451993085022, 0.45158720239473, 0.49986603548352]),
    },
    'M062X': {
        -1: np.array([-0.24076315072104, 0.02405308720508, 0.06776493317329]),
        0: np.array([-0.00444384692897, 0.02885897898585, 0.16747914479167]),
        1: np.array([0.95071992515633, 0.97504949683333, 0.99916253449809]),
    },
    'CAM-B3LYP': {
        -1: np.array([-0.22656603258900, 0.02401590266477, 0.07550155022876]),
        0: np.array([-0.00361563353820, 0.04311309241302, 0.17628925722949]),
        1: np.array([0.97274168121312, 0.99449611819119, 1.01391420069302]),
    },
}


class KnownValues(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        mol = gto.Mole()
        mol.verbose = 0
        mol.output = '/dev/null'
        mol.atom = '''
        O                  0.64372820    0.14077399   -0.04477253
        O                 -0.64862595   -0.12779073   -0.05445498
        H                  1.16027512   -0.65947800    0.36730132
        H                 -1.12109306    0.55561188    0.42651873
        '''
        mol.charge = 0
        mol.spin = 2
        mol.basis = '631g'
        mol.symmetry = True
        cls.mol = mol.build()

    @classmethod
    def tearDownClass(cls):
        cls.mol.stdout.close()

    def _check_functional(self, xc, frozen, averaged=False):
        mf = self.mol.ROKS(xc=xc)
        if averaged:
            mf = mf.average_occ()
        mf.run()
        self.assertTrue(mf.converged)
        for delta_s, ref in REFS[xc].items():
            with self.subTest(xc=xc, deltaS=delta_s):
                td = nttda.NTTDA(mf, frozen=frozen)
                td.deltaS = delta_s
                td.kernel()
                self.assertTrue(np.all(td.converged))
                np.testing.assert_allclose(td.e, ref, atol=1e-6, rtol=0)

                energies = np.linalg.eigvalsh(td.get_ab())
                if delta_s == -1:
                    # Remove the redundant OO identity, the sole zero mode in this example.
                    energies = np.delete(energies, np.argmin(abs(energies)))
                np.testing.assert_allclose(td.e, energies[:td.nstates], atol=1e-6, rtol=0)

    # This molecule has closed orbitals 0-7, open orbitals 8-9, and LUMO 10.
    # Use one frozen selection and one SCF reference per functional.
    def test_hf_nttda(self):
        # Freeze both core orbitals and two valence occupied orbitals.
        self._check_functional('HF', frozen=4)

    def test_svwn_nttda(self):
        # Freeze the same occupied orbitals and the LUMO, using AOCSCF.
        self._check_functional('SVWN', frozen=[0, 1, 2, 3, 10], averaged=True)

    def test_m062x_nttda(self):
        # A noncontiguous occupied selection, plus LUMO and LUMO+1.
        self._check_functional('M062X', frozen=[0, 2, 4, 10, 11])

    def test_cam_b3lyp_nttda(self):
        # Keep only two closed orbitals; also freeze LUMO and LUMO+1.
        self._check_functional('CAM-B3LYP', frozen=[0, 1, 2, 3, 4, 5, 10, 11], averaged=True)


if __name__ == '__main__':
    print('Full tests for frozen-orbital NTTDA')
    unittest.main()
