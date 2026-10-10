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

"""ROKS orbital Hessian and Z-vector response for NTTDA gradients."""

import numpy as np

from pyscf import dft, lib


def canonical_pairs(tdobj, compact=True):
    """Canonical spatial-orbital rotations and their ROKS residual type."""
    occ = np.asarray(tdobj._scf.mo_occ)
    closed = np.flatnonzero(occ == 2)
    open_ = np.flatnonzero(occ == 1)
    virtual = np.flatnonzero(occ == 0)
    pairs = []
    if not compact:
        for indices, name in (
                (closed, "cc"), (open_, "oo"), (virtual, "vv")):
            for p_local in range(1, len(indices)):
                for q_local in range(p_local):
                    pairs.append((indices[p_local], indices[q_local], name))
    pairs.extend((o, c, "co") for o in open_ for c in closed)
    pairs.extend((v, c, "cv") for v in virtual for c in closed)
    pairs.extend((v, o, "ov") for v in virtual for o in open_)
    return tuple(pairs)


def _spin_focks_mo(mf):
    fock = mf.get_fock()
    mo = np.asarray(mf.mo_coeff)
    return mo.conj().T @ fock.focka @ mo, mo.conj().T @ fock.fockb @ mo


def _response_reference(mf):
    if (isinstance(mf, dft.KohnShamDFT)
            and mf._numint._xc_type(mf.xc) != "HF"):
        reference = mf.to_uks()
        reference.verbose = 0
        return reference
    return mf


def _make_adjoint_action(tdobj, pairs):
    """Cache reference operators for the full ROKS adjoint matrix."""
    mf = tdobj._scf
    mo = np.asarray(mf.mo_coeff)
    occ = np.asarray(mf.mo_occ)
    fock_alpha, fock_beta = _spin_focks_mo(mf)
    occupation_alpha = (occ > 0).astype(float)
    occupation_beta = (occ == 2).astype(float)
    response = _response_reference(mf).gen_response(hermi=1)

    def apply_one(vector):
        source_alpha, source_beta = _unpack_zvector_source(tdobj, pairs, vector)
        gradient = fock_alpha @ (source_alpha + source_alpha.T)
        gradient += fock_beta @ (source_beta + source_beta.T)
        density_alpha = mo @ source_alpha @ mo.conj().T
        density_beta = mo @ source_beta @ mo.conj().T
        density_alpha = 0.5 * (density_alpha + density_alpha.T)
        density_beta = 0.5 * (density_beta + density_beta.T)
        potential_alpha, potential_beta = response(
            np.asarray((density_alpha, density_beta))
        )
        potential_alpha = mo.conj().T @ potential_alpha @ mo
        potential_beta = mo.conj().T @ potential_beta @ mo
        gradient += potential_alpha * occupation_alpha[None, :]
        gradient += potential_alpha.T * occupation_alpha[None, :]
        gradient += potential_beta * occupation_beta[None, :]
        gradient += potential_beta.T * occupation_beta[None, :]
        return gradient

    return apply_one


def make_hessian_transpose_action(tdobj, pairs=None):
    """Return a matrix-free action for the transpose ROKS Hessian."""
    if pairs is None:
        pairs = canonical_pairs(tdobj, compact=True)
    adjoint = _make_adjoint_action(tdobj, pairs)

    def apply(vector):
        vector = np.asarray(vector)
        if vector.ndim == 1:
            return pack_m_matrix(adjoint(vector), pairs)
        return np.asarray([pack_m_matrix(adjoint(row), pairs) for row in vector])

    return apply, pairs


def pack_m_matrix(matrix, pairs):
    antisymmetric = matrix - matrix.T
    return np.asarray([antisymmetric[p, q] for p, q, _name in pairs])


def _preconditioner(tdobj, pairs):
    fock_alpha, fock_beta = _spin_focks_mo(tdobj._scf)
    epsilon_alpha = np.diag(fock_alpha)
    epsilon_beta = np.diag(fock_beta)
    epsilon_common = 0.5 * (epsilon_alpha + epsilon_beta)
    diagonal = []
    for p, q, name in pairs:
        if name in ("cc", "oo", "vv"):
            value = epsilon_common[p] - epsilon_common[q]
        elif name == "co":
            value = epsilon_beta[p] - epsilon_beta[q]
        elif name == "cv":
            value = (
                epsilon_alpha[p] - epsilon_alpha[q]
                + epsilon_beta[p] - epsilon_beta[q]
            )
        elif name == "ov":
            value = epsilon_alpha[p] - epsilon_alpha[q]
        else:
            raise ValueError("unknown ROKS pair type %s" % name)
        diagonal.append(value)
    diagonal = np.asarray(diagonal)
    small = np.abs(diagonal) < 1e-8
    diagonal[small] = np.where(diagonal[small] < 0.0, -1e-8, 1e-8)
    return diagonal


def _solve_zvector(action, diagonal, rhs, tolerance=1e-12, max_cycle=None):
    """Preconditioned Krylov solve with a check of the actual equation.

    Krylov's trial-vector linear-dependence cutoff can terminate before
    the residual converges, especially after rotations within C/O/V spaces.
    Up to two scaled residual corrections recover the requested accuracy;
    ``max_cycle`` bounds each Krylov solve.
    """
    if rhs.size == 0:
        return np.empty_like(rhs)
    initial = rhs / diagonal
    if max_cycle is None:
        max_cycle = len(rhs)

    def operator(vector):
        vector = np.asarray(vector)
        if vector.ndim == 1:
            return action(vector) / diagonal - vector
        return np.asarray([action(row) / diagonal - row for row in vector])

    def solve(preconditioned_rhs):
        return np.asarray(lib.krylov(
            operator,
            preconditioned_rhs,
            tol=tolerance,
            max_cycle=max_cycle,
            lindep=1e-22,
            hermi=False,
            verbose=0,
        )).reshape(-1)

    solution = solve(initial)
    # Respect the requested residual up to an absolute roundoff floor.
    threshold = max(tolerance, 1e-12)
    for correction in range(3):
        residual = rhs - action(solution)
        error = np.max(np.abs(residual), initial=0.0)
        if np.isfinite(error) and error <= threshold:
            return solution
        if not np.isfinite(error) or correction == 2:
            raise RuntimeError('NTTDA orbital response did not converge (residual %.3e)' % error)
        preconditioned = residual / diagonal
        scale = np.linalg.norm(preconditioned)
        solution += scale * solve(preconditioned / scale)

def solve_zvector(action, pairs, tdobj, rhs, tolerance=1e-12, max_cycle=None):
    """Solve the ROKS adjoint with its spin-resolved preconditioner."""
    return _solve_zvector(action, _preconditioner(tdobj, pairs), rhs,
                          tolerance, max_cycle)


def _unpack_zvector_source(tdobj, pairs, zvector):
    nmo = tdobj._scf.mo_coeff.shape[1]
    source_alpha = np.zeros((nmo, nmo))
    source_beta = np.zeros_like(source_alpha)
    for value, (p, q, name) in zip(zvector, pairs):
        if name in ("cc", "oo", "vv"):
            source_alpha[p, q] += 0.5 * value
            source_beta[p, q] += 0.5 * value
        elif name == "co":
            source_beta[p, q] += value
        elif name == "cv":
            source_alpha[p, q] += value
            source_beta[p, q] += value
        elif name == "ov":
            source_alpha[p, q] += value
        else:
            raise ValueError("unknown ROKS pair type %s" % name)
    return source_alpha, source_beta


def zvector_adjoint_matrix(tdobj, pairs, zvector):
    """Full MO adjoint matrix satisfying ``z.H(kappa)=Tr(G.T kappa)``."""
    return _make_adjoint_action(tdobj, pairs)(zvector)


def zvector_probe_densities(tdobj, pairs, zvector):
    mo = np.asarray(tdobj._scf.mo_coeff)
    source_alpha, source_beta = _unpack_zvector_source(
        tdobj, pairs, zvector,
    )
    return (
        mo @ source_alpha @ mo.conj().T,
        mo @ source_beta @ mo.conj().T,
    )
