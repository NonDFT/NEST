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

"""The AOCSCF kitchen sink: X2C, density fitting, and three solver choices.

Recommended order: average_occ -> sfx2c1e -> density_fit -> solver.
Set the reference, Hamiltonian and J/K approximation before the solver.
Choose one of newton(), FR(), or SGM() as the last step.
"""

from pyscf import gto
from nest import aocscf  # necessary import
from nest.deltascf import fr, sgm  # necessary imports

mol = gto.M(
    atom='O 0 0 0; H 0 -.757 .587; H 0 .757 .587',
    basis='6-31g', spin=0)
ground = mol.RKS(xc='B3LYP').sfx2c1e().density_fit().run()

# FR/SGM need initial orbitals and the target occupations.
# Promote one HOMO electron to the LUMO: occupations 2/0 become 1/1.
occupations = ground.mo_occ.copy()
occupations[4:6] = 1
triplet = mol.copy()
triplet.spin = 2

# 1. Average-occupation reference -> X2C Hamiltonian -> DF J/K -> Newton.
# Newton can start from the default SCF guess.
mf = triplet.ROKS(xc='B3LYP').average_occ().sfx2c1e().density_fit()
mf_newton = mf.newton().run()
print(f'Newton converged: {mf_newton.converged}')
print(f'Average-occupation energy: {mf_newton.e_avg_occ:.12f} Ha')
print(f'High-spin energy: {mf_newton.e_tot:.12f} Ha')

# 2. Apply FR to the configured reference. FR is an alternative to Newton.
mf = triplet.ROKS(xc='B3LYP').average_occ().sfx2c1e().density_fit()
mf_fr = mf.FR().run(ground.mo_coeff, occupations)
print(f'FR converged: {mf_fr.converged}')
print(f'Average-occupation energy: {mf_fr.e_avg_occ:.12f} Ha')
print(f'High-spin energy: {mf_fr.e_tot:.12f} Ha')

# 3. Apply SGM to the configured reference as another solver choice.
mf = triplet.ROKS(xc='B3LYP').average_occ().sfx2c1e().density_fit()
mf_sgm = mf.SGM().run(ground.mo_coeff, occupations)
print(f'SGM converged: {mf_sgm.converged}')
print(f'Average-occupation energy: {mf_sgm.e_avg_occ:.12f} Ha')
print(f'High-spin energy: {mf_sgm.e_tot:.12f} Ha')

# AOCSCF unwraps only a temporary reference to report the high-spin energy.
# mf_sgm itself still carries SGM. NTTDA energies accept either object;
# undo_sgm() gives subsequent calculations the ordinary SCF driver.
reference = mf_sgm.undo_sgm()
