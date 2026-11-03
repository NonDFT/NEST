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

"""Optimize orbitals with average occupations and report the high-spin energy."""

from pyscf import gto
from nest import aocscf  # noqa: F401 - registers ROKS.average_occ()

mol = gto.M(
    atom="""
    H  0.000000  0.934473 -0.588078
    H  0.000000 -0.934473 -0.588078
    C  0.000000  0.000000  0.000000
    O  0.000000  0.000000  1.221104
    """,
    basis="6-31g",
    spin=2,
    symmetry=True
)
mf = mol.ROKS(xc="SVWN").average_occ().run()

print(f"SCF converged: {mf.converged}")
print(f"Average-occupation SCF energy: {mf.e_avg_occ:.12f} Ha")
print(f"High-spin energy: {mf.e_tot:.12f} Ha")
mf.analyze(verbose=4)  # same as other SCF methods
