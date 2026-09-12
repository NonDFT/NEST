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

'''Run a restricted open-shell Kohn-Sham (ROKS) calculation using the SGM'''
# Q-Chem reference input for this CH4 core excitation benchmark:
#
# $rem
# METHOD BHHLYP
# BASIS aug-cc-pvtz
# SYMMETRY false
# SYM_IGNORE true
# NO_REORIENT True
# XC_GRID 000075000302
# UNRESTRICTED false
# delta_scf true
# $end
#
# $delta_scf
#   triplet restricted
#   triplet_SCF_algorithm SGM
#   somo_1 1
#   somo_2 6
# $end
#
# Q-Chem output:
#   Restricted open-shell triplet state = -29.947480 Ha
#   SOMO(1): best initial orbital 5 overlap 0.999002
#   SOMO(2): best initial orbital 6 overlap 0.879378
#   Excitation energy = 10.554376 Ha / 287.199215 eV

from pyscf import gto
from nest.soscf import sgm


atom = '''
C 0.00000000 0.00000000 0.00000000
H 0.62760000 0.62760000 0.62760000
H 0.62760000 -0.62760000 -0.62760000
H -0.62760000 0.62760000 -0.62760000
H -0.62760000 -0.62760000 0.62760000
'''

mol = gto.M(atom=atom, charge=0, spin=0, basis='aug-cc-pvtz')
mf = mol.RKS(xc='BHandHLYP')
mf.grids.atom_grid = (75, 302)
mf.kernel()

setocc = mf.to_uks().mo_occ
setocc[1][0] -= 1
setocc[0][5] += 1
ro_occ = setocc[0] + setocc[1]

mol1 = gto.M(atom=atom, charge=0, spin=2, basis='aug-cc-pvtz')
mf1 = mol1.ROKS(xc='BHandHLYP')
mf1.grids.atom_grid = (75, 302)
mf1.mo_coeff = mf.mo_coeff
mf1.mo_occ = ro_occ

sgm_mf = mf1.SGM()
sgm_mf.verbose = 4
sgm_mf.kernel()

print('SGM converged =', sgm_mf.converged)
print('SGM energy    = %.12f Ha' % sgm_mf.e_tot)
