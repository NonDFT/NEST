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

"""Run deltaS=-1 NTTDA with SGM/FR delta-SCF references for CH4.

Optimize the core-excited triplet with SGM or FR, then use it as the
reference for singlet NTTDA. Use the CH4 geometry and C 1s/LUMO guess
from 01_sgm.py and 02_fr.py, with a 6-31G* basis and the default grid.
"""

import numpy as np
from pyscf import gto
from pyscf.data import nist
from nest import nttda  # Registers NTTDA on ROKS.
from nest.deltascf import fr, sgm

# 1. Calculate the closed-shell ground state.
mol = gto.M(
    atom="""
    C  0.0000  0.0000  0.0000
    H  0.6276  0.6276  0.6276
    H  0.6276 -0.6276 -0.6276
    H -0.6276  0.6276 -0.6276
    H -0.6276 -0.6276  0.6276
    """,
    basis='631G*',
    spin=0,
    symmetry=False,
)
ground = mol.RKS(xc='BHandHLYP')
ground.kernel()

# 2. Make a triplet guess by changing two occupations from 2/0 to 1/1.
# Indices start at zero: orbital 0 is C 1s, orbital 5 is the ground-state LUMO.
hole_index = 0
particle_index = 5
initial_orbitals = ground.mo_coeff.copy()
initial_occupations = ground.mo_occ.copy()
initial_occupations[hole_index] = 1
initial_occupations[particle_index] = 1

# 3. Optimize the delta-SCF triplet reference with SGM or FR.
triplet_mol = mol.copy()
triplet_mol.spin = 2

triplet_scf = triplet_mol.ROKS(xc=ground.xc)
excited = fr.FR(triplet_scf)  # or sgm.SGM(triplet_scf)
excited.verbose = 4
excited.kernel(initial_orbitals, initial_occupations)
excited.analyze(verbose=4)

# 4. Run deltaS=-1 NTTDA on the optimized delta-SCF reference.
reference = excited  # for SGM, reference = excited.undo_sgm()
td = reference.NTTDA()
td.deltaS = -1  # S_reference=1 -> S_final=0 (singlets).
td.nstates = 10  # Include core-excited singlets above the lower valence roots.
td.kernel()
td.analyze(verbose=4)

# Supplementary excitation-energy comparison.
# E_triplet - E_RKS defines the delta-SCF gap for the settings used here.
# omega is relative to the triplet, so add that gap for a common RKS zero.
delta_scf_ev = (reference.e_tot - ground.e_tot) * nist.HARTREE2EV
nttda_rks_ev = delta_scf_ev + td.e * nist.HARTREE2EV
nttda_excitation_ev = (td.e - td.e[0]) * nist.HARTREE2EV
print(f'delta-SCF triplet excitation = {delta_scf_ev:.6f} eV')
