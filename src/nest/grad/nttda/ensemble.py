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

"""Average-occupation orbital response for AOCSCF NTTDA gradients."""

import numpy as np

from pyscf import dft, lib
from pyscf.scf import hf

from .roks import finish_gradient, pack_m_matrix, _solve_zvector


def canonical_pairs(tdobj):
    """Independent rotations between unequal-occupation orbital spaces."""
    occupation = np.asarray(tdobj._scf.mo_occ)
    labels = {2: "c", 1: "o", 0: "v"}
    rows, columns = np.where(hf.uniq_var_indices(occupation))
    return tuple(
        (int(p), int(q), labels[int(occupation[q])] + labels[int(occupation[p])])
        for p, q in zip(rows, columns)
    )


def _rotation_matrix(vector, pairs, nmo):
    rotation = np.zeros((nmo, nmo))
    for value, (p, q, _name) in zip(vector, pairs):
        rotation[p, q] += value
        rotation[q, p] -= value
    return rotation


def _weighted_source(tdobj, pairs, vector):
    occupation = np.asarray(tdobj._scf.mo_occ)
    nmo = occupation.size
    source = np.zeros((nmo, nmo))
    for value, (p, q, _name) in zip(vector, pairs):
        source[p, q] += value * (occupation[q] - occupation[p])
    return source


def _fock_mo(mf):
    orbitals = np.asarray(mf.mo_coeff)
    return orbitals.conj().T @ np.asarray(mf.get_fock()) @ orbitals


def make_hessian_transpose_action(tdobj, pairs=None):
    """Return the symmetric average-occupation orbital Hessian action."""
    mf = tdobj._scf
    orbitals = np.asarray(mf.mo_coeff)
    occupation = np.asarray(mf.mo_occ)
    occupation_difference = occupation[None, :] - occupation[:, None]
    nmo = orbitals.shape[1]
    if pairs is None:
        pairs = canonical_pairs(tdobj)
    fock = _fock_mo(mf)
    response = lib.view(mf, dft.rks.RKS).gen_response(hermi=1)

    def apply_one(vector):
        rotation = _rotation_matrix(vector, pairs, nmo)
        density_mo = rotation * occupation_difference
        density_ao = orbitals @ density_mo @ orbitals.conj().T
        potential = response(density_ao)
        potential_mo = orbitals.conj().T @ potential @ orbitals
        fock_derivative = (
            fock @ rotation - rotation @ fock + potential_mo
        )
        return np.asarray([
            occupation_difference[p, q] * fock_derivative[p, q]
            for p, q, _name in pairs
        ])

    def apply(vector):
        vector = np.asarray(vector)
        if vector.ndim == 1:
            return apply_one(vector)
        return np.asarray([apply_one(row) for row in vector])

    return apply, pairs


def zvector_adjoint_matrix(tdobj, pairs, zvector):
    """Return the full coefficient derivative of ``z . g_orbital``."""
    mf = tdobj._scf
    orbitals = np.asarray(mf.mo_coeff)
    occupation = np.asarray(mf.mo_occ)
    source = _weighted_source(tdobj, pairs, zvector)
    fock = _fock_mo(mf)
    gradient = fock @ (source + source.T)

    density = orbitals @ source @ orbitals.conj().T
    density = 0.5 * (density + density.conj().T)
    potential = lib.view(mf, dft.rks.RKS).gen_response(hermi=1)(density)
    potential = orbitals.conj().T @ potential @ orbitals
    gradient += potential * occupation[None, :]
    gradient += potential.conj().T * occupation[None, :]
    return gradient


def zvector_probe_densities(tdobj, pairs, zvector):
    """Spin probes for the nuclear derivative of the common ensemble Fock."""
    orbitals = np.asarray(tdobj._scf.mo_coeff)
    source = _weighted_source(tdobj, pairs, zvector)
    total = orbitals @ source @ orbitals.conj().T
    return 0.5 * total, 0.5 * total


def _preconditioner(tdobj, pairs):
    occupation = np.asarray(tdobj._scf.mo_occ)
    epsilon = np.diag(_fock_mo(tdobj._scf))
    diagonal = np.asarray([
        (occupation[q] - occupation[p]) * (epsilon[p] - epsilon[q])
        for p, q, _name in pairs
    ])
    small = np.abs(diagonal) < 1e-8
    diagonal[small] = np.where(diagonal[small] < 0.0, -1e-8, 1e-8)
    return diagonal


def solve_zvector(action, pairs, tdobj, rhs, tolerance=1e-12, max_cycle=None):
    """Solve the average-occupation adjoint with its occupation-weighted diagonal."""
    return _solve_zvector(action, _preconditioner(tdobj, pairs), rhs,
                          tolerance, max_cycle)


__all__ = [
    "canonical_pairs",
    "finish_gradient",
    "make_hessian_transpose_action",
    "pack_m_matrix",
    "solve_zvector",
    "zvector_adjoint_matrix",
    "zvector_probe_densities",
]
