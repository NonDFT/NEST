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
from pyscf import fci, gto
from nest.deltascf import fr, rks
from nest.nttda import NTTDA
from nest.nttda.nttda import _orbital_indices


class KnownValues(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        mol = gto.Mole()
        mol.verbose = 0
        mol.output = '/dev/null'
        mol.atom = """
        C  0.0000  0.0000  0.0000
        H  0.6776  0.6076  0.6576
        H  0.6076 -0.6676 -0.6176
        H -0.6476  0.6576 -0.6076
        H -0.5976 -0.6176  0.6476
        """
        mol.basis = '631g'
        cls.mol = mol.build()

    @classmethod
    def tearDownClass(cls):
        cls.mol.stdout.close()

    def test_bhandhlyp_rks_transition(self):
        ground = self.mol.RKS(xc='BHandHLYP').run()
        triplet = self.mol.copy()
        triplet.spin = 2
        occupations = ground.mo_occ.copy()
        occupations[0] = occupations[5] = 1
        reference = fr.FR(triplet.ROKS(xc=ground.xc))
        reference.max_cycle = 100
        reference.kernel(ground.mo_coeff, occupations)
        self.assertTrue(ground.converged and reference.freeze_converged and reference.converged)
        td = reference.NTTDA().set(deltaS=-1, nstates=10, max_cycle=100).run()
        self.assertTrue(np.all(td.converged))

        states = np.array([6, 7, 8])
        length = rks.transition_properties(td, ground, state=states)
        velocity = rks.transition_properties(td, ground, state=states, gauge='velocity')
        # RKS and lowest-NTTDA energy zeros, then both gauges for each initial state.
        actual = np.column_stack((
            length['excitation_energy'], td.e[states - 1] - td.e[0],
            length['oscillator_strength'], velocity['oscillator_strength'],
            td.oscillator_strength(state=states), td.oscillator_strength(state=states, gauge='velocity'),
        ))
        expected = np.array([
            [10.712278470040, 10.788020457156, 0.000871055425, 0.000786624652, 0.001142923885, 0.001065660415],
            [10.748238647355, 10.823980634470, 0.040416228826, 0.037064944885, 0.030114708832, 0.029810493892],
            [10.757011472634, 10.832753459749, 0.039844029975, 0.036747569163, 0.030580935131, 0.030493267562],
        ])
        self.assertAlmostEqual(abs(actual - expected).max(), 0, delta=1e-5)

    def test_transition_against_fci(self):
        # Four electrons in independent MO sets; expand both states in the
        # same full-CI space and evaluate the one-body density with PySCF.
        mol = gto.M(atom='H 0 0 0; H .2 0 1.3; H 1.4 .1 0; H 1.1 .3 1.5',
                    basis='631g', verbose=0)
        ground = mol.RKS()
        triplet = mol.copy()
        triplet.spin = 2
        mf = triplet.ROKS()
        overlap_ao = ground.get_ovlp()
        values, vectors = np.linalg.eigh(overlap_ao)
        mf.mo_coeff = (vectors / np.sqrt(values)) @ vectors.T
        rng = np.random.default_rng(17)
        ground.mo_coeff = mf.mo_coeff @ np.linalg.qr(rng.normal(size=overlap_ao.shape))[0]
        nmo = mol.nao
        ground.mo_occ = np.array([2, 2] + [0] * (nmo - 2))
        mf.mo_occ = np.array([1, 2, 1] + [0] * (nmo - 3))
        ground.e_tot, mf.e_tot = -2., -1.5
        td = NTTDA(mf)
        td.deltaS = -1
        cs, os, vs = _orbital_indices(td)
        amplitudes = rng.normal(size=(3, len(cs) + 2, len(vs) + 2))
        for root in amplitudes:
            root[len(cs):, :2] -= np.eye(2) * np.trace(root[len(cs):, :2]) / 2
        amplitudes = np.linalg.qr(amplitudes.reshape(3, -1).T)[0].T.reshape(amplitudes.shape)
        td.xy = [(root, 0) for root in amplitudes]
        td.e = np.array([-.7, .2, .4])

        strings = list(map(int, fci.cistring.make_strings(range(nmo), 2)))
        address = {s: i for i, s in enumerate(strings)}
        csfs = list(rks._singlet_csfs(cs, os, vs, nmo))
        expansion = np.zeros((len(csfs), len(strings), len(strings)))
        for row, csf in enumerate(csfs):
            for det, coefficient in csf.items():
                alpha, beta = det & ((1 << nmo) - 1), det >> nmo
                expansion[row, address[alpha], address[beta]] = coefficient
        ci = np.einsum('ri,iab->rab', amplitudes.reshape(3, -1), expansion)
        ct = mf.mo_coeff
        transform = ct.T @ overlap_ao @ ground.mo_coeff[:, :2]
        coefficients = [np.linalg.det(transform[[p for p in range(nmo) if (s >> p) & 1]]) for s in strings]
        bra = np.outer(coefficients, coefficients)

        masses = mol.atom_mass_list()
        origin = masses @ mol.atom_coords() / masses.sum()
        with mol.with_common_orig(origin):
            length_ao = mol.intor_symmetric('int1e_r', comp=3)
        velocity_ao = -mol.intor('int1e_ipovlp', comp=3)
        nuclear = mol.atom_charges() @ (mol.atom_coords() - origin)
        for gauge, operator in (('length', length_ao), ('velocity', velocity_ao)):
            actual = rks.transition_properties(td, ground, gauge=gauge)
            operator_mo = np.einsum('up,xuv,vq->xpq', ct, operator, ct)
            for i, ket in enumerate(ci):
                overlap = np.vdot(bra, ket)
                gamma = fci.direct_spin1.trans_rdm1(bra, ket, nmo, (2, 2)).T
                moment = np.einsum('pq,xpq->x', gamma, operator_mo)
                self.assertAlmostEqual(actual['overlap'][i], overlap, delta=1e-12)
                if gauge == 'length':
                    moment -= nuclear * overlap
                    key, factor = 'dipole', td.e[i] + .5
                else:
                    key, factor = 'velocity_dipole', 1 / (td.e[i] + .5)
                self.assertAlmostEqual(abs(actual[key][i] - moment).max(), 0, delta=1e-12)
                self.assertAlmostEqual(actual['oscillator_strength'][i],
                                       (2 / 3) * factor * np.dot(moment, moment), delta=1e-12)


if __name__ == '__main__':
    print('Full tests for RKS-to-NTTDA transition properties')
    unittest.main()
