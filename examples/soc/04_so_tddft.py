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

"""Compare state-interaction SOC with direct SOC-TDA for H2O2."""

import numpy as np
from pyscf import gto
from pyscf.data.nist import HARTREE2EV

from nest.soc import SOTDA, TDRHFSOC


mol = gto.M(
    atom="""
    O   0.64372820   0.14077399  -0.04477253
    O  -0.64862595  -0.12779073  -0.05445498
    H   1.16027512  -0.65947800   0.36730132
    H  -1.12109306   0.55561188   0.42651873
    """,
    basis="631g",
    charge=0,
    spin=0,
    symmetry=False,
    verbose=0,
)
mf = mol.RKS(xc="PBE").run()

nocc = np.count_nonzero(mf.mo_occ == 2)
nvir = np.count_nonzero(mf.mo_occ == 0)
nov = nocc * nvir
dimension = 1 + 4 * nov
print(f"nocc={nocc}, nvir={nvir}, nov={nov}, SOC dimension={dimension}")

# Conventional route: solve every scalar singlet and triplet TDA state, then
# couple the complete state space with SOC.  Keeping all nov roots makes this
# a unitary change of basis rather than a truncated state-interaction model.
singlet = mf.TDA().set(singlet=True, nstates=nov, max_cycle=300).run()
triplet = mf.TDA().set(singlet=False, nstates=nov, max_cycle=300).run()
state_interaction = TDRHFSOC(
    singlet,
    triplet,
    soctype="SOMF",
    include_reference=True,
).run()

# Direct route: diagonalize the complete [reference,S,T-1,T0,T+1] SOC-TDA
# Casida matrix.  xy[root][0] is a one-dimensional complex vector of length
# 1 + 4*nov when include_reference=True.
direct = SOTDA(
    mf,
    soctype="SOMF",
    include_reference=True,
).set(
    nstates=dimension,
    max_cycle=300,
).run()

error = np.max(np.abs(direct.e - state_interaction.e))
print(f"Maximum difference over all {dimension} roots: {error:.3e} Hartree")

np.set_printoptions(precision=8, suppress=True)
print("Lowest 10 energies relative to the lowest root / eV")
print(np.column_stack((
    (state_interaction.e[:10] - state_interaction.e[0]) * HARTREE2EV,
    (direct.e[:10] - direct.e[0]) * HARTREE2EV,
)))
