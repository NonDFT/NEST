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

"""C1 CH4: NR/sfX2C1e FR, sfX2C1e SGM, and both oscillator-strength gauges.

Run with OMP_NUM_THREADS=4 python 06_sfx2c1e_transition.py.
No nuclear gradients are used. Apply sfx2c1e before wrapping FR or SGM.
sfX2C1e uses PySCF's one-electron scalar-relativistic Hamiltonian with
nonrelativistic two-electron interactions; property operators include their
one-electron picture change by default. It is not a full relativistic
many-electron response treatment, and length/velocity agreement is not exact.
"""

import numpy as np
from pyscf import gto
from pyscf.data import nist
from nest.deltascf import fr, sgm
from nest.nttda.rks import transition_properties

mol = gto.M(
    atom='''C  0.0000  0.0000  0.0000
    H  0.6776  0.6076  0.6576
    H  0.6076 -0.6676 -0.6176
    H -0.6476  0.6576 -0.6076
    H -0.5976 -0.6176  0.6476''',
    basis='6-31g*', spin=0, symmetry=False,
)
triplet_mol = mol.copy()
triplet_mol.spin = 2
references = {}
for relativistic in (False, True):
    ground = mol.RKS(xc='BHandHLYP')
    mf = triplet_mol.ROKS(xc=ground.xc)
    if relativistic:
        ground = ground.sfx2c1e()
        mf = mf.sfx2c1e()
    ground.chkfile = None
    ground.conv_tol = 1e-11
    ground.conv_tol_grad = 1e-7
    ground.kernel()
    assert ground.converged
    occupations = ground.mo_occ.copy()
    occupations[0] = occupations[5] = 1
    reference = fr.FR(mf)
    reference.chkfile = None
    reference.conv_tol = 1e-11
    reference.conv_tol_grad = 1e-7
    reference.max_cycle = 100
    reference.kernel(ground.mo_coeff, occupations)
    assert reference.freeze_converged and reference.converged
    references['sfX2C-FR' if relativistic else 'NR-FR'] = ground, reference

# SGM can reach a different stationary state from an unrelaxed ground-MO
# guess. Here the converged NR core-triplet orbitals supply a target-specific
# guess for an independent sfX2C SGM optimization. Check the SOMO overlap.
nr_reference = references['NR-FR'][1]
x2c_ground, x2c_fr = references['sfX2C-FR']
optimized = sgm.SGM(triplet_mol.ROKS(xc=x2c_ground.xc).sfx2c1e())
optimized.chkfile = None
optimized.conv_tol_grad = 1e-7
optimized.kernel(nr_reference.mo_coeff, nr_reference.mo_occ)
assert optimized.converged
x2c_sgm = optimized.undo_sgm()  # Retains with_x2c and the relativistic hcore.
somo_overlap = (x2c_fr.mo_coeff[:, x2c_fr.mo_occ == 1].T @ x2c_fr.get_ovlp()
                @ x2c_sgm.mo_coeff[:, x2c_sgm.mo_occ == 1])
assert np.linalg.svd(somo_overlap, compute_uv=False).min() > .99999
assert abs(x2c_fr.e_tot - x2c_sgm.e_tot) < 1e-9
references['sfX2C-SGM'] = x2c_ground, x2c_sgm

for name, (ground, reference) in references.items():
    td = reference.NTTDA()
    td.chkfile = None
    td.deltaS = -1
    td.nstates = 30
    td.conv_tol = 1e-7
    td.lindep = 1e-18
    td.max_cycle = 100
    td.kernel()
    assert np.all(td.converged)
    states = np.flatnonzero(abs(td.e * nist.HARTREE2EV) < 30)[:10] + 1
    length = transition_properties(td, ground, state=states)
    velocity = transition_properties(td, ground, state=states, gauge='velocity')
    print(f'\n{name}: RKS -> NTTDA, E_ROKS - E_RKS = {(reference.e_tot - ground.e_tot) * nist.HARTREE2EV:.6f} eV')
    print('root   E/eV       f_length       f_velocity')
    for i, state in enumerate(states):
        print(f'{state:3d}  {length["excitation_energy"][i] * nist.HARTREE2EV:9.4f}'
              f'  {length["oscillator_strength"][i]:.7e}  {velocity["oscillator_strength"][i]:.7e}')
    if name.startswith('sfX2C'):
        # Same orbitals/energies, bare integrals vs transformed operators.
        # picture_change=False is a diagnostic, not the sfX2C velocity.
        bare_length = transition_properties(td, ground, state=states, picture_change=False)
        bare_velocity = transition_properties(td, ground, state=states, gauge='velocity', picture_change=False)
        print('Picture change, relative change of first-peak f (length / velocity):',
              length['oscillator_strength'][0] / bare_length['oscillator_strength'][0] - 1,
              velocity['oscillator_strength'][0] / bare_velocity['oscillator_strength'][0] - 1)
        shifted = transition_properties(td, ground, state=states, origin=[1.0, -.5, .2], gauge='velocity')
        np.testing.assert_allclose(shifted['dipole'], length['dipole'], atol=1e-10, rtol=0)
        np.testing.assert_allclose(shifted['velocity_dipole'], velocity['velocity_dipole'], atol=1e-10, rtol=0)
        # The same operators are used for transitions between NTTDA roots.
        print('NTTDA-root f_length:', td.oscillator_strength(state=states))
        print('NTTDA-root f_velocity:', td.oscillator_strength(state=states, gauge='velocity'))
