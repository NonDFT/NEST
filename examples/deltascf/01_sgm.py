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

"""CH4 core excitation with ordinary SGM and a canonical-orbital guess.

This is a restricted triplet calculation.
Only use the excitation energy when the excited-state calculation converges.
"""
# Q-Chem reference input for this CH4 core excitation benchmark:
#
# $rem
# METHOD BHHLYP
# BASIS aug-cc-pvtz
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
#   somo_1 1
#   somo_2 6
# $end
#
# Q-Chem output:
#   Restricted open-shell triplet state = -29.947480 Ha
#   SOMO(1): best initial orbital 5 overlap 0.999002
#   SOMO(2): best initial orbital 6 overlap 0.879378
#   Excitation energy = 10.554376 Ha / 287.199215 eV

from pyscf import gto
from pyscf.data import nist
from nest.deltascf import sgm

# 1. Calculate the closed-shell ground state.
mol = gto.M(
    atom="""
    C  0.0000  0.0000  0.0000
    H  0.6276  0.6276  0.6276
    H  0.6276 -0.6276 -0.6276
    H -0.6276  0.6276 -0.6276
    H -0.6276 -0.6276  0.6276
    """,
    basis='aug-cc-pvtz',
    spin=0,
    symmetry=False,
)
ground = mol.RKS(xc='BHandHLYP')
ground.grids.atom_grid = (75, 302)
ground.kernel()

# 2. Make a triplet guess by changing two occupations from 2/0 to 1/1.
# Indices start at zero: orbital 0 is C 1s, orbital 5 is the ground-state LUMO.
hole_index = 0
particle_index = 5
initial_orbitals = ground.mo_coeff.copy()
initial_occupations = ground.mo_occ.copy()
initial_occupations[hole_index] = 1
initial_occupations[particle_index] = 1

# 3. Optimize the excited-state orbitals using the original SGM preconditioner.
triplet_mol = mol.copy()
triplet_mol.spin = 2  # PySCF spin = N_alpha - N_beta = 2S.
triplet_scf = triplet_mol.ROKS(xc=ground.xc)
triplet_scf.grids.atom_grid = (75, 302)
excited = sgm.SGM(triplet_scf)
excited.verbose = 4
excited.kernel(initial_orbitals, initial_occupations)

print('SGM converged:', excited.converged)
print('Last energy (Ha):', excited.e_tot)
if excited.converged:
    excitation_energy = (excited.e_tot - ground.e_tot) * nist.HARTREE2EV
    print('Triplet excitation energy (eV):', excitation_energy)
