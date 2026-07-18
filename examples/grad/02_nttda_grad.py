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

"""Calculate an analytic NTTDA excited-state gradient with NEST."""

from pyscf import gto

from nest import nttda  # noqa: F401  # Registers ROKS.NTTDA().


mol = gto.M(
    atom='Li 0 0 0; H 0 0 1.60',
    basis='sto-3g',
    spin=0,  # 2S = N_alpha - N_beta; here the ROKS reference has S_i = 0.
    unit='Bohr',
    verbose=4,
)

mf = mol.ROKS(xc='PBE').set(
    conv_tol=1e-12,
    conv_tol_grad=1e-9,
    max_cycle=200,
)
mf.grids.level = 3
mf.kernel()
if not mf.converged:
    raise RuntimeError('ROKS reference did not converge')

td = mf.NTTDA().set(
    deltaS=1,  # S_f = S_i + deltaS; supported values: -1, 0, +1.
    nobeta=False,
    nstates=3,
    conv_tol=1e-9,
    max_cycle=200,
).run()
if not all(td.converged):
    raise RuntimeError('one or more NTTDA roots did not converge')

# NTTDA gradient roots are one-based: state=1 corresponds to td.e[0].
td_grad = td.Gradients().set(
    cphf_conv_tol=1e-10,
)

# Total excited-state gradient d(E_ROKS + omega_state)/dR in Eh/Bohr.
total_gradient = td_grad.kernel(state=1)
print('Total NTTDA excited-state gradient (Eh/Bohr):')
print(total_gradient)

# Optional: excitation-energy derivative d(omega_state)/dR only.
excitation_gradient = td_grad.grad_elec(td.xy[0])
print('Excitation-energy derivative only (Eh/Bohr):')
print(excitation_gradient)

# Optional: calculate only selected atoms; atom indices are zero-based.
# selected_gradient = td.Gradients().kernel(state=1, atmlst=[0, 2])
