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

'''
Analytic nuclear gradient of the Dz0SCF (average-occupation) reference.

Dz0SCF drives one set of orbitals with average occupations 2/1/0 for a
high-spin open-shell reference, and takes the high-spin ROKS energy evaluated
on those orbitals as the reference energy.  ``nuc_grad_method()`` returns the
analytic gradient of that reference energy, which is the zero state used by
the NTTDA excited-state gradients (see examples/nttda/02_nttda_dz0scf_grad.py).

The reference is non-stationary on the average-occupation orbitals, so the
driver solves a Z-vector equation for the orbital response; a diffuse enough
integration grid is required for the force sum to vanish.
'''

from pyscf import gto
from nest import dz0scf  # necessary import
from nest.dz0scf import DZ0SCF

atom = '''
N   0.000000  -0.040000   0.000000
H   0.000000   0.780000   0.590000
H   0.000000  -0.860000   0.520000
'''
mol = gto.M(atom=atom, charge=0, spin=1, basis='6-31g', verbose=3)
fun = 'PBE'  # try also 'SVWN', 'B3LYP', 'M06-2X', etc.
mf = DZ0SCF(mol, xc=fun)
mf.conv_tol = 1e-12
mf.conv_tol_grad = 1e-9
mf.max_cycle = 120
mf.grids.level = 5      # dense grid: the force sum is grid-sensitive
mf.grids.prune = None
mf.small_rho_cutoff = 0.0
mf.kernel()

print('Dz0SCF reference energy: %.12f' % mf.high_spin_energy())

grad = mf.nuc_grad_method().kernel()
print('Analytic reference gradient (Eh/Bohr):\n', grad)
print('Force sum (should be ~0):\n', grad.sum(axis=0))

# Gradients can also be restricted to selected atoms:
grad_n = mf.nuc_grad_method().kernel(atmlst=[0])
print('Gradient on the nitrogen atom only:\n', grad_n)
