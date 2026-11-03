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

"""Run NTTDA with frozen core orbitals and calculate an excited-state gradient."""

from pyscf import gto

from nest import nttda  # noqa: F401 - registers ROKS.NTTDA()

mol = gto.M(
    atom="""
    H  0.000000  0.934473 -0.588078
    H  0.000000 -0.934473 -0.588078
    C  0.000000  0.000000  0.000000
    O  0.000000  0.000000  1.221104
    """,
    basis="6-31g",
    spin=2,
)
mf = mol.ROKS(xc="PBE").run()

# Freeze the C 1s and O 1s orbitals in the excitation space.
# All orbitals are still included in the SCF reference.
td = mf.NTTDA(frozen=2).run()
td.analyze()
print("Total energies (Ha):", td.e_tot)
print("A matrix shape:", td.get_ab().shape)

gradient = td.Gradients().kernel(state=1)
print("State-1 gradient (Eh/Bohr):\n", gradient)

# Other choices: explicit MO indices (zero-based), or automatic core selection.
# Here -1 also freezes the last virtual orbital.
# td = mf.NTTDA(frozen=[0, 1, -1]).run()
# td = mf.NTTDA().set_frozen(method="auto").run()
