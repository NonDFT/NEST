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

'''
Analytic gradient spin-flip TDDFT/TDA examples.
'''

import numpy as np
from pyscf import gto
from nest import sftda  # necessary import

atom = '''
H  0.000000  0.934473 -0.588078
H  0.000000 -0.934473 -0.588078
C  0.000000  0.000000  0.000000
O  0.000000  0.000000  1.221104
'''
mol = gto.M(atom=atom, charge=0, spin=2, basis='6-31g', verbose=3)
fun = 'CAM-B3LYP'  # try also 'TPSS', 'SVWN', etc.
mf = mol.UKS(xc=fun)
mf.grids.level = 7  # increase grid size for accuracy
mf.kernel()
td = mf.SFTDDFT()  # mf.SFTDA() for SF-TDA
td.extype = 1
td.collinear_samples = 20  # use collinear='col' for collinear SF-TDDFT
td.nstates = 5
td.kernel()

tdg = td.Gradients()
anal_grad = tdg.kernel(state=1)  # 1 for first excited state

def numerical_gradient(f, mol, delta=1e-5):
    coords = mol.atom_coords()
    grad = np.zeros_like(coords)
    for i in range(mol.natm):
        for j in range(3):
            orig_val = coords[i, j]
            coords[i, j] = orig_val + delta
            mol.set_geom_(coords, unit='Bohr')
            f_plus = f(mol)
            coords[i, j] = orig_val - delta
            mol.set_geom_(coords, unit='Bohr')
            f_minus = f(mol)
            grad[i, j] = (f_plus - f_minus) / (2 * delta)
            coords[i, j] = orig_val
    mol.set_geom_(coords, unit='Bohr')
    return grad

def f(mol):
    mf = mol.UKS(xc=fun)
    mf.kernel()
    td = mf.SFTDDFT()
    td.extype = 1
    td.nstates = 5
    td.collinear_samples = 20
    td.kernel()
    return td.e[0] + mf.e_tot

num_grad = numerical_gradient(f, mol)

print("Analytic Gradient:\n", anal_grad)
print("Numerical Gradient:\n", num_grad)
print(np.allclose(anal_grad, num_grad, atol=1e-5))
