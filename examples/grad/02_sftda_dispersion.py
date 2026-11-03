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
SF-TDA total energies and gradients with D3/D4 dispersion.
Required dependencies: pyscf-dispersion
"""

from pyscf import gto
from nest import sftda  # Register SFTDA/SFTDDFT on PySCF reference objects.

mol = gto.M(
    atom="O 0 0 0; H 0 -0.757 0.587; H 0 0.757 0.587",
    basis="sto-3g",
    spin=2,
    verbose=3,
)
mf = mol.UKS(xc="PBE")
# Set dispersion BEFORE SCF. Parameters are selected using mf.xc.
# 'd3bj' or 'd3bj2b': D3(BJ), two-body only
# 'd3bjatm':          D3(BJ), including the ATM three-body term
# 'd4':              D4, including the ATM three-body term
mf.disp = "d4"
mf.kernel()

# The same gradient interface is available for mf.SFTDDFT().
td = mf.SFTDA().set(nstates=2, collinear="col").run()
print("TD roots relative to the reference (Hartree):", td.e)
print("Total state energies including dispersion (Hartree):", td.e_tot)

# state=1 means the first TD root, which can lie below the SCF reference.
# state=0 requests the SCF reference gradient instead.
grad = td.Gradients().kernel(state=1)
print("First-root total gradient including dispersion (Hartree/Bohr):")
print(grad)

# For geometry optimization, the scanner returns consistent total energy and
# gradient and recomputes dispersion when the geometry changes.
# It reuses the previous SCF density and TD amplitudes as initial guesses.
# The root index does not track state character.
scanner = td.Gradients().as_scanner(state=1)
energy, gradient = scanner(mol)
print("Scanner total energy (Hartree):", energy)
# mf.e_tot, td.e_tot, and the total gradients already include dispersion.
