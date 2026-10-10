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
Build the dense NTTDA A matrix and compare with Davidson.

Use a small system: get_ab stores a dense excitation matrix and the full
four-index reference-kernel tensor in the MO basis.
"""

import numpy as np
from pyscf import gto

from nest import nttda
from nest.nttda.get_ab import get_ab

mol = gto.M(
    atom="O 0 0 0; H 0 0 1; H 0 1 0",
    basis="sto-3g",
    spin=2,
    verbose=0,
)
mf = mol.ROKS(xc="CAM-B3LYP").run(conv_tol=1e-11)
assert mf.converged

for delta_s in (-1, 0, 1):
    td = mf.NTTDA().set(deltaS=delta_s, nstates=2)
    a = get_ab(mf, deltaS=delta_s, nobeta=td.nobeta)  # Also available as td.get_ab().

    if delta_s == -1:
        # A uses the flattened [[CO, CV], [OO, OV]] amplitude layout.
        # Remove the known OO identity direction by changing basis, rather
        # than discarding every eigenvalue near zero.
        nc = np.count_nonzero(mf.mo_occ == 2)
        no = np.count_nonzero(mf.mo_occ == 1)
        nv = np.count_nonzero(mf.mo_occ == 0)
        q = np.zeros((nc + no, no + nv))
        q[nc:, :no] = np.eye(no) / np.sqrt(no)
        basis = np.linalg.qr(q.reshape(-1, 1), mode="complete")[0][:, 1:]
        energies = np.linalg.eigvalsh(basis.T @ a @ basis)
    else:
        # deltaS=0: CO(1), CV(1), scalar OO(1), OV(1), CV(0).
        # deltaS=+1: flattened CV amplitudes only.
        energies = np.linalg.eigvalsh(a)

    td.run()
    assert np.all(td.converged)
    np.testing.assert_allclose(energies[:td.nstates], td.e, atol=1e-8, rtol=0)
    print(f"deltaS={delta_s:+d}, A shape={a.shape}")
    print("Dense energies (Hartree):   ", energies[:td.nstates])
    print("Davidson energies (Hartree):", td.e)
