#!/usr/bin/env python
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

"""C1 CH4: independent RKS -> core-excited singlet NTTDA dipoles.

Run with, for example, OMP_NUM_THREADS=4 python 05_rks_transition.py.
The ROKS core-LUMO triplet is optimized by FR. Its orbitals differ from
the RKS orbitals; the new interface handles their nonorthogonality.
No X2C, frozen-core constraint, or extra root orthogonalization is used.
"""

import numpy as np
from pyscf import gto
from pyscf.data import nist

from nest.deltascf import fr
from nest.nttda.rks import transition_properties

mol = gto.M(
    atom='''
    C  0.0000  0.0000  0.0000
    H  0.6776  0.6076  0.6576
    H  0.6076 -0.6676 -0.6176
    H -0.6476  0.6576 -0.6076
    H -0.5976 -0.6176  0.6476
    ''',
    basis='6-31g*', spin=0, symmetry=False,
)
# Place the center of mass at zero for the electronic-only comparison.
# The total dipole of this neutral system is invariant to this choice.
masses = mol.atom_mass_list()
center = masses @ mol.atom_coords() / masses.sum()
mol.set_geom_(mol.atom_coords() - center, unit='Bohr')
ground = mol.RKS(xc='BHandHLYP')
ground.chkfile = None
ground.conv_tol = 1e-11
ground.conv_tol_grad = 1e-7
ground.kernel()
assert ground.converged

# Zero-based MO indices: C 1s -> LUMO.
occupations = ground.mo_occ.copy()
occupations[0] = occupations[5] = 1
triplet_mol = mol.copy()
triplet_mol.spin = 2
reference = fr.FR(triplet_mol.ROKS(xc=ground.xc))
reference.chkfile = None
reference.conv_tol = 1e-11
reference.conv_tol_grad = 1e-7
reference.max_cycle = 100
reference.kernel(ground.mo_coeff, occupations)
assert reference.freeze_converged and reference.converged
# With SGM instead, pass excited.undo_sgm() to NTTDA.

td = reference.NTTDA()
# To omit excitations out of all doubly occupied orbitals instead:
# td = reference.NTTDA(frozen=np.flatnonzero(reference.mo_occ == 2).tolist())
# Frozen occupied orbitals still contribute to overlap and dipole integrals.
# SOMOs must remain active: this does not strictly enforce a core hole.
td.chkfile = None
td.deltaS = -1
td.nstates = 30  # Includes several roots that refill the core hole.
# Tight enough for the dipole comparison without chasing numerical noise
# in a large block of nearly degenerate roots.
td.conv_tol = 1e-7
td.lindep = 1e-18
td.max_cycle = 100
td.kernel()
assert np.all(td.converged)
td.analyze()

# For this example the refilled-core roots have omega near -270 to -293 eV.
# Keep the first ten roots within 30 eV of the core-excited triplet. This is
# an example-specific energy filter, not a general core-hole classifier;
# inspect td.analyze() when adapting the calculation to another system.
states = np.flatnonzero(np.abs(td.e * nist.HARTREE2EV) < 30)[:10] + 1
properties = transition_properties(td, ground, state=states)

# Existing interface: lowest computed NTTDA singlet -> the selected roots.
# These are different initial states. Compare norms since phases are arbitrary.
old_dipole = td.transition_dipole(ref=1, state=states)
old_strength = td.oscillator_strength(ref=1, state=states)
print('\nDipoles in ea0; new excitation energies use E_ROKS + omega - E_RKS.')
print('root   E/eV       overlap      |d_NT|     |r_RKS|    |d_RKS,total|    f_NT        f_RKS')
for i, state in enumerate(states):
    print(f'{state:3d}  {properties["excitation_energy"][i] * nist.HARTREE2EV:9.4f}'
          f'  {properties["overlap"][i]: .3e}'
          f'  {np.linalg.norm(old_dipole[i]):10.6f}'
          f'  {np.linalg.norm(properties["electronic_dipole"][i]):10.6f}'
          f'  {np.linalg.norm(properties["dipole"][i]):13.6f}'
          f'  {old_strength[i]:.6e}  {properties["oscillator_strength"][i]:.6e}')

# The fixed nuclear operator contributes nuclear_dipole * overlap. NEST's
# positive-r sign convention returns the negative of the physical total mu.
# For neutral CH4 it cancels electronic origin dependence exactly. Charged
# systems with nonzero overlap retain origin dependence and emit a warning.
shifted = transition_properties(td, ground, state=states, origin=[1.0, -0.5, 0.2])
np.testing.assert_allclose(shifted['dipole'], properties['dipole'], atol=1e-10, rtol=0)
print('Neutral-system origin-shift check passed.')
