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
NTTDA excited-state gradients on an AOCSCF (average-occupation) reference.

An AOCSCF reference gives a common average-occupation orbital set for every
spin channel.  NTTDA built on that reference can target the same-spin channel
(``deltaS=0``) and the spin-lowering channel (``deltaS=-1``); the total state
energy is the high-spin reference energy plus the NTTDA excitation energy, so
its gradient is the sum of the reference gradient and the excitation-energy
gradient.

``td.Gradients().kernel(state=n)`` returns the analytic gradient of
``E_reference + omega_n`` for state ``n`` (1 for the lowest root); ``state=0``
returns the reference gradient.  Analytic gradients are available for
``deltaS = -1`` and ``0``; ``deltaS = +1`` uses ``method='finite_diff'``.
'''

from pyscf import gto
from nest import aocscf, nttda  # necessary imports

atom = '''
C    0.020000  -0.030000   0.010000
H   -0.020000   0.800000   0.620000
H    0.030000  -0.910000   0.500000
'''
mol = gto.M(atom=atom, charge=0, spin=2, basis='sto-3g', verbose=3)
fun = 'B3LYP'
mf = mol.ROKS(xc=fun).average_occ()
mf.conv_tol = 1e-12
mf.conv_tol_grad = 1e-9
mf.max_cycle = 150
mf.grids.level = 5      # dense grid: the force sum is grid-sensitive
mf.grids.prune = None
mf.small_rho_cutoff = 0.0
mf.kernel()

for delta_s in (-1, 0):
    td = mf.NTTDA().set(
        deltaS=delta_s,     # Sf = Si + deltaS
        nstates=3,
        conv_tol=1e-9,
        max_cycle=200,
        verbose=0,
    ).run()
    print('deltaS = %+d' % delta_s)
    print('  NTTDA excitation energies:', td.e)
    print('  total energies (E_ref + omega):', td.total_energies())

    grad = td.Gradients().kernel(state=1)
    print('  state-1 analytic gradient (Eh/Bohr):\n', grad)
    print('  force sum (should be ~0):\n', grad.sum(axis=0))

    ref_grad = td.Gradients().kernel(state=0)
    print('  reference (state-0) gradient:\n', ref_grad)
    print('  excitation-only contribution (state 1 - state 0):\n',
          grad - ref_grad)
