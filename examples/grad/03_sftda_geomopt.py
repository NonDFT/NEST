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

"""
Optimize the first SF-TDA root with geomeTRIC and D4 dispersion.
Required dependencies: geometric, pyscf-dispersion
"""

from pyscf import gto
from pyscf.geomopt import geometric_solver
from nest import sftda  # Register SF-TDA on PySCF reference objects.

mol = gto.M(
    atom="O 0 0 0; H 0 -0.757 0.587; H 0 0.757 0.587",
    basis="6-31g",
    spin=2,
)
mf = mol.UKS(xc="B3LYP").set(disp="d4")

# The TD constructor runs SCF if needed; the optimizer triggers the TD calculation.
td = mf.SFTDA().set(extype=1, collinear='mcol', nstates=3)

# It reuses the SCF density and projects old TD amplitudes onto the new MO basis.
# A fixed root index does not track electronic character through root crossings.
scanner = td.Gradients().as_scanner(state=1)
optimizer = geometric_solver.GeometryOptimizer(scanner)
optimizer.max_cycle = 50
mol_eq = optimizer.kernel()

print("Geometry optimization converged:", optimizer.converged)
print("Optimized geometry (Angstrom):")
print(mol_eq.tostring(format="xyz"))
