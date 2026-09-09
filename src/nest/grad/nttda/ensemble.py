"""Average-occupation orbital response for Dz0SCF NTTDA gradients."""

import numpy as np

from pyscf import lib
from pyscf.scf import hf

from .roks import GradientComponents, _orbital_gradient, pack_m_matrix


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
    response = mf.gen_response(hermi=1)

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
    potential = mf.gen_response(hermi=1)(density)
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


def solve_zvector(action, pairs, tdobj, rhs, tolerance=1e-12,
                  max_cycle=None):
    """Solve the average-occupation orbital adjoint equation."""
    diagonal = _preconditioner(tdobj, pairs)
    initial = rhs / diagonal
    if max_cycle is None:
        max_cycle = len(rhs)

    def operator(vector):
        vector = np.asarray(vector)
        if vector.ndim == 1:
            return action(vector) / diagonal - vector
        return np.asarray([action(row) / diagonal - row for row in vector])

    solution = lib.krylov(
        operator,
        initial,
        tol=tolerance,
        max_cycle=max_cycle,
        lindep=1e-22,
        hermi=False,
        verbose=0,
    )
    return np.asarray(solution).reshape(-1)


def finish_gradient(
        gradient_driver, tdobj, m_matrix, direct, atmlst,
        tolerance, max_cycle, fock_direct, direct_fock_probes=None):
    """Solve the Dz0SCF average-occupation adjoint and assemble ``d omega / dR``."""
    transpose_action, pairs = make_hessian_transpose_action(tdobj)
    rhs = pack_m_matrix(m_matrix, pairs)
    zvector = solve_zvector(
        transpose_action,
        pairs,
        tdobj,
        rhs,
        tolerance=tolerance,
        max_cycle=max_cycle,
    )
    adjoint = zvector_adjoint_matrix(tdobj, pairs, zvector)
    residual = float(np.max(np.abs(pack_m_matrix(adjoint, pairs) - rhs)))
    probe_alpha, probe_beta = zvector_probe_densities(
        tdobj, pairs, zvector,
    )
    if direct_fock_probes is None:
        fock_contraction = fock_direct(
            gradient_driver, tdobj, probe_alpha, probe_beta, atmlst=atmlst,
        )
        direct_total = direct
    else:
        direct_alpha, direct_beta = direct_fock_probes
        fock_contractions = fock_direct(
            gradient_driver,
            tdobj,
            np.asarray((direct_alpha, probe_alpha)),
            np.asarray((direct_beta, probe_beta)),
            atmlst=atmlst,
        )
        direct_total = direct + fock_contractions[0]
        fock_contraction = fock_contractions[1]
    orbital = _orbital_gradient(
        tdobj, m_matrix, adjoint, fock_contraction, atmlst=atmlst,
    )
    return GradientComponents(
        m_matrix=m_matrix,
        direct=direct_total,
        orbital=orbital,
        total=direct_total + orbital,
        zvector=zvector,
        residual=residual,
    )


__all__ = [
    "canonical_pairs",
    "finish_gradient",
    "make_hessian_transpose_action",
    "pack_m_matrix",
    "solve_zvector",
    "zvector_adjoint_matrix",
    "zvector_probe_densities",
]
