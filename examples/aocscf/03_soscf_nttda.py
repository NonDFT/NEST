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

"""Run NTTDA from a Newton-optimized average-occupation ROKS reference."""

from pyscf import gto
from nest import aocscf, nttda  # necessary imports

mol = gto.M(
    atom='O 0 0 0; H 0 -.757 .587; H 0 .757 .587',
    basis='6-31g', spin=2)

# Newton accepts the usual default SCF guess.
mf = mol.ROKS(xc='B3LYP').average_occ()
mf.conv_tol = 1e-10
mf.conv_tol_grad = 1e-6
mf = mf.newton().run()
print(f'Newton converged: {mf.converged}')
print(f'Average-occupation energy: {mf.e_avg_occ:.12f} Ha')
print(f'High-spin energy: {mf.e_tot:.12f} Ha')

# Remove the Newton driver, retaining the AOCSCF orbitals and energies.
mf = mf.remove_soscf()
td = mf.NTTDA().set(nstates=5, deltaS=-1).run()
td.analyze(verbose=4)
print(f'NTTDA total energies (Ha): {td.e_tot}')
