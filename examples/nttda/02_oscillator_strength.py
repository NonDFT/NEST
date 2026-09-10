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

"""Length-gauge transition dipoles and oscillator strengths within each spin sector."""

from pyscf import gto
from nest import nttda  # Registers NTTDA on ROKS.

mol = gto.M(
    atom="""
    O  0.64372820  0.14077399 -0.04477253
    O -0.64862595 -0.12779073 -0.05445498
    H  1.16027512 -0.65947800  0.36730132
    H -1.12109306  0.55561188  0.42651873
    """,
    spin=2,
    basis='sto-3g',
)
mf = mol.ROKS(xc='SVWN').run()

for deltaS in (-1, 0, 1):
    td = mf.NTTDA().set(deltaS=deltaS, nstates=4).run()
    print(f'Final spin S = {mol.spin / 2 + deltaS}')
    # Indices are 1-based computed roots in this spin sector.
    # ref=1 is its lowest computed root, not the ROKS reference state.
    print('Transition dipoles (a.u.), 1 -> [2, 3]:')
    print(td.transition_dipole(ref=1, state=[2, 3]))
    print('Oscillator strengths, 1 -> all other computed roots:')
    print(td.oscillator_strength(ref=1))
    print('Oscillator strength, 1 -> 2 (scalar):')
    print(td.oscillator_strength(ref=1, state=2))
    # Downward transitions retain the negative energy difference.
    print('Oscillator strength, 2 -> 1:')
    print(td.oscillator_strength(ref=2, state=1))
