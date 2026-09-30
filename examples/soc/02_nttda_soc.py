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

"""Run an NTTDA SOC calculation and print the complete SOC analysis."""

from pyscf import gto
from nest import nttda  # noqa: F401  # Registers the NTTDA methods.

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

# 1. Spin-restricted high-spin scalar reference.
mf = mol.ROKS(xc="SVWN").run()

# 2. NTTDA states in two spin sectors (deltaS=+1 is also supported).
td0 = mf.NTTDA().set(deltaS=0, nstates=1).run()
tdm1 = mf.NTTDA().set(deltaS=-1, nstates=2).run()

# 3. Build and diagonalize the SOC Hamiltonian.
#
# The PySCF-style entry point is recommended.  This first driver combines the
# calculated deltaS=0 and deltaS=-1 states without an artificial reference.
soc_without_reference = td0.SOC(
    tdm1, soctype="SOMF", include_reference=False,
).run()

# When only deltaS=-1 states are supplied, the scalar reference can be inserted
# explicitly.  It is never included by default.
soc_with_reference = tdm1.SOC(
    soctype="SOMF", include_reference=True,
).run()

# Equivalent explicit constructors:
#
#     from nest.soc import NTTDASOC
#     soc_without_reference = NTTDASOC(
#         td0, tdm1, soctype="SOMF", include_reference=False,
#     ).run()
#     soc_with_reference = NTTDASOC(
#         tdm1, soctype="SOMF", include_reference=True,
#     ).run()
#
# A list can be passed instead of positional NTTDA objects:
#
#     soc_without_reference = NTTDASOC(
#         [td0, tdm1], soctype="SOMF", include_reference=False,
#     ).run()
#
# Empty construction is also supported.  Inputs and options are read only when
# run()/kernel() starts:
#
#     soc_without_reference = NTTDASOC()
#     soc_without_reference.tds = [td0, tdm1]
#     soc_without_reference.soctype = "SOMF"
#     soc_without_reference.include_reference = False
#     e, v = soc_without_reference.kernel()
#
# The same setup using StreamObject.run keyword arguments:
#
#     soc_without_reference = NTTDASOC().run(
#         tds=[td0, tdm1], soctype="SOMF", include_reference=False,
#     )
#
# Available soctype values are "SOMF", "SOMF_AMFI", "Zeff", "1e", "X2C1E",
# "X2CAMF", and "X2CMP".  X2CAMF and X2CMP require the optional socutils package.

# 4. Print scalar states, SOC blocks, SOCCs, SOC energies, and compositions.
print("\nSOC without the scalar reference")
soc_without_reference.analyze(verbose=4)
print("\nSOC with the scalar reference")
soc_with_reference.analyze(verbose=4)

# The numerical results remain available for further processing:
# driver.states        spin-free input states
# driver.state_slices  mapping from spin-free states to Hamiltonian slices
# driver.h_soc         complex Hermitian SOC Hamiltonian, in Hartree
# driver.e, .v         SOC eigenvalues and eigenvectors
