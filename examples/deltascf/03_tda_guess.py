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

"""C2H3F C1 core excitation: a core-TDA guess followed by SGM or FR.

The chosen TDA root defines an initial guess; it does not guarantee the
identity of the final SCF solution. Check convergence before using energies.
"""

import numpy as np
from scipy.linalg import null_space
from pyscf import gto, tdscf
from pyscf.data import nist
from nest.deltascf import fr, sgm

# 1. Calculate the closed-shell ground state.
mol = gto.M(
    atom="""
    C1  0.000000  -0.246412  -1.271068
    C2  0.000000   0.457081  -0.154735
    F1  0.000000  -0.119195   1.052878
    H1  0.000000   0.272328  -2.210194
    H2  0.000000  -1.319906  -1.249847
    H3  0.000000   1.530323  -0.095954
    """,
    basis={'C1': 'aug-cc-pvtz', 'C2': 'aug-cc-pvtz',
           'F1': 'cc-pvtz', 'H': 'cc-pvdz'},
    spin=0,
    symmetry=False,
)
ground = mol.RKS(xc='BHandHLYP')
ground.kernel()

# 2. Obtain a particle orbital from single-hole, triplet core-TDA.
# These zero-based indices apply to THIS molecule and basis.
hole_index = 2      # C1 1s orbital.
particle_index = 12 # First ground-state virtual; also the slot for our particle orbital.
tda_root = 1       # Second TDA root: the even-parity root with a large LUMO component.

# Allow excitations only from the chosen core orbital, to all virtual orbitals.
# This freezing applies to TDA only, not to the subsequent SCF calculation.
frozen_occupied = []
for orbital_index in range(particle_index):
    if orbital_index != hole_index:
        frozen_occupied.append(orbital_index)

tda = tdscf.TDA(ground)
tda.singlet = False
tda.frozen = frozen_occupied
tda.kernel(nstates=3)

for root_index, (energy, amplitudes) in enumerate(zip(tda.e, tda.xy)):
    excitation_amplitudes, _ = amplitudes
    particle_vector = excitation_amplitudes[0].copy()
    particle_vector /= np.linalg.norm(particle_vector)
    lumo_weight = particle_vector[0]**2
    print(f'TDA root {root_index}: {energy * nist.HARTREE2EV:.6f} eV, '
          f'ground-state LUMO weight = {lumo_weight:.4f}')

# Only one occupied orbital participates, so X has a single row. Its entries
# are the coefficients of the particle orbital in the OLD virtual basis.
# Normalize explicitly because PySCF's restricted TDA uses 2 * ||X||^2 = 1.
excitation_amplitudes, _ = tda.xy[tda_root]
particle_vector = excitation_amplitudes[0].copy()
particle_vector /= np.linalg.norm(particle_vector)

# Put the particle orbital in the original LUMO slot. Complete the remaining
# virtual columns with orthonormal combinations perpendicular to that vector.
# These other columns are a basis completion, not higher TDA excited states.
old_virtual_orbitals = ground.mo_coeff[:, particle_index:]
remaining_virtual_vectors = null_space(particle_vector.reshape(1, -1))
initial_orbitals = ground.mo_coeff.copy()
initial_orbitals[:, particle_index] = old_virtual_orbitals @ particle_vector
initial_orbitals[:, particle_index + 1:] = old_virtual_orbitals @ remaining_virtual_vectors

initial_occupations = ground.mo_occ.copy()
initial_occupations[hole_index] = 1
initial_occupations[particle_index] = 1

# 3. Optimize the restricted triplet. Choose FR or SGM on the next line.
triplet_mol = mol.copy()
triplet_mol.spin = 2
triplet_scf = triplet_mol.ROKS(xc=ground.xc)
excited = fr.FR(triplet_scf)  # Or: sgm.SGM(triplet_scf)
excited.verbose = 4
excited.conv_tol_grad = 1e-5
# FR first freezes both singly occupied orbitals, then releases them with IMOM.
# verbose=4 prints both stage boundaries and the final SOMO-subspace overlaps.
excited.conv_tol = 1e-9
excited.max_cycle = 200
excited.kernel(initial_orbitals, initial_occupations)

print('Optimizer converged:', excited.converged, 'cycles:', excited.cycles)
print('Last energy (Ha):', excited.e_tot)
if excited.converged:
    excitation_energy = (excited.e_tot - ground.e_tot) * nist.HARTREE2EV
    print('Triplet excitation energy (eV):', excitation_energy)
