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

"""C1 CH4: core-excited NTTDA singlets and transitions from the lowest root."""

import numpy as np
from pyscf import gto
from pyscf.data import nist
from nest.deltascf import fr

mol = gto.M(
    atom="""
    C  0.0000  0.0000  0.0000
    H  0.6776  0.6076  0.6576
    H  0.6076 -0.6676 -0.6176
    H -0.6476  0.6576 -0.6076
    H -0.5976 -0.6176  0.6476
    """,
    basis='6-31g*',
    spin=0,
    symmetry=False,
)
ground = mol.RKS(xc='BHandHLYP').run()

# Optimize the C 1s -> LUMO triplet. MO indices start at zero.
occupations = ground.mo_occ.copy()
occupations[0] = occupations[5] = 1
triplet = mol.copy()
triplet.spin = 2
reference = fr.FR(triplet.ROKS(xc=ground.xc))
reference.max_cycle = 100
reference.kernel(ground.mo_coeff, occupations)

td = reference.NTTDA()
td.deltaS = -1
td.nstates = 15  # Includes the lower roots that refill the core hole.
td.max_cycle = 100
td.kernel()
td.analyze()

# For this CH4 reference, core-excited roots lie within 30 eV of the triplet;
# the refilled-core roots have omega around -290 eV. Root numbers are 1-based.
states = np.flatnonzero(abs(td.e * nist.HARTREE2EV) < 30) + 1

# ref=1 is the lowest NTTDA singlet, not the triplet SCF reference.
energy = (td.e[states - 1] - td.e[0]) * nist.HARTREE2EV
length = td.oscillator_strength(ref=1, state=states)
velocity = td.oscillator_strength(ref=1, state=states, gauge='velocity')
print('root   E_NTTDA/eV    f_length       f_velocity')
for i, state in enumerate(states):
    print(f'{state:3d}  {energy[i]:11.4f}  {length[i]:.6e}  {velocity[i]:.6e}')
