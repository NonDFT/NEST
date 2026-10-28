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

"""Run NTTDA from FR and SGM average-occupation delta-SCF references."""

from pyscf import gto
from nest import aocscf, nttda  # necessary imports
from nest.deltascf import fr, sgm  # necessary imports

mol = gto.M(
    atom='O 0 0 0; H 0 -.757 .587; H 0 .757 .587',
    basis='6-31g', spin=0)
ground = mol.RKS(xc='B3LYP').run()

# Promote one HOMO electron to the LUMO, producing two singly occupied orbitals.
occupations = ground.mo_occ.copy()
occupations[4:6] = 1
triplet = mol.copy()
triplet.spin = 2

# FR freezes the two SOMOs, then releases all orbitals with IMOM.
mf = triplet.ROKS(xc='B3LYP').average_occ()
mf_fr = mf.FR().run(ground.mo_coeff, occupations)
print(f'FR converged: {mf_fr.converged}')
print(f'Average-occupation energy: {mf_fr.e_avg_occ:.12f} Ha')
print(f'High-spin energy: {mf_fr.e_tot:.12f} Ha')

# remove_soscf() leaves references without a Newton driver unchanged.
reference = mf_fr.remove_soscf()
td_fr = reference.NTTDA().set(nstates=5, deltaS=-1).run()
td_fr.analyze(verbose=4)
print(f'FR + NTTDA total energies (Ha): {td_fr.e_tot}')

# SGM minimizes the squared orbital gradient with fixed occupations.
mf = triplet.ROKS(xc='B3LYP').average_occ()
mf_sgm = mf.SGM().run(ground.mo_coeff, occupations)
print(f'SGM converged: {mf_sgm.converged}')
print(f'Average-occupation energy: {mf_sgm.e_avg_occ:.12f} Ha')
print(f'High-spin energy: {mf_sgm.e_tot:.12f} Ha')

# SGM has its own wrapper, removed with undo_sgm().
reference = mf_sgm.undo_sgm().remove_soscf()
td_sgm = reference.NTTDA().set(nstates=5, deltaS=-1).run()
td_sgm.analyze(verbose=4)
print(f'SGM + NTTDA total energies (Ha): {td_sgm.e_tot}')
