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

"""Run an SF-TDA SOC calculation and print the complete SOC analysis."""

from pyscf import gto
from nest import sftda  # noqa: F401  # Registers the SFTDA methods.

mol = gto.M(
    atom="""
    O   0.64372820   0.14077399  -0.04477253
    O  -0.64862595  -0.12779073  -0.05445498
    H   1.16027512  -0.65947800   0.36730132
    H  -1.12109306   0.55561188   0.42651873
    """,
    basis="631g",
    charge=0,
    spin=2,
    symmetry=False,
)

# 1. High-spin scalar reference.
mf = mol.UKS(xc="SVWN").run()

# 2. Spin-flip-down TDA states. SOC currently supports extype=1 only.
td = mf.SFTDA().set(
    extype=1,
    nstates=3,
    collinear="mcol",
    collinear_samples=50,
).run()

# 3. Build and diagonalize the SOMF SOC Hamiltonian.
soc = td.SOC(soctype="SOMF").run()

# 4. Print scalar states, SOC blocks, SOCCs, SOC energies, and compositions.
# verbose=4 enables the detailed block matrices and eigenstate compositions.
soc.analyze(verbose=4)

# The numerical results remain available for further processing:
# soc.states      spin-free input states
# soc.state_slices  mapping from spin-free states to Hamiltonian slices
# soc.h_soc       complex Hermitian SOC Hamiltonian, in Hartree
# soc.e, soc.v    SOC eigenvalues and eigenvectors
