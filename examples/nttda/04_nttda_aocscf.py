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

"""Run NTTDA from an average-occupation ROKS reference."""

from pyscf import gto

from nest import aocscf, nttda  # noqa: F401 - registers ROKS.average_occ() and NTTDA()

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
mf = mol.ROKS(xc="CAM-B3LYP").average_occ()
mf.kernel()
td = mf.NTTDA().set(nstates=5, deltaS=-1)
# nobeta does not work for NTTDA with a average-occupation ROKS reference
td.kernel()

td.analyze(verbose=4)
print(f"Total energies (Ha): {td.e_tot}", end=",\n")
print(f"which equals to mf.e_tot + td.e: {mf.e_tot + td.e}")
