# Copyright 2026 The NEST Developers. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Small-system overlap checks, independent of the analytic NAC contractions.

Determinant bits are alpha first, then beta, with spatial orbitals ordered
C/O/V. The MO overlap is reordered accordingly, even for noncontiguous
occupation spaces. No determinant expansion is used by the analytic driver.
The lowering/same-spin construction follows the BSCC NTTDA NAC reference;
the scalar CV configuration uses NEST's opposite phase.
"""

from functools import lru_cache

import numpy as np
from pyscf import gto
from scipy.optimize import linear_sum_assignment

from nest.nttda.nttda import _orbital_indices, _split_amplitudes


def _combine(parts):
    state = {}
    for weight, component in parts:
        for determinant, coefficient in component.items():
            state[determinant] = state.get(determinant, 0.0) + weight * coefficient
    return {det: c for det, c in state.items() if abs(c) > 1e-14}


def _one_body(state, create, annihilate):
    """Apply a_create^dagger a_annihilate, including the fermionic sign."""
    result = {}
    for determinant, coefficient in state.items():
        if not determinant & (1 << annihilate):
            continue
        intermediate = determinant ^ (1 << annihilate)
        if intermediate & (1 << create):
            continue
        parity = (determinant & ((1 << annihilate) - 1)).bit_count()
        parity += (intermediate & ((1 << create) - 1)).bit_count()
        result[intermediate | (1 << create)] = (-1) ** parity * coefficient
    return result


def _normalize(state):
    norm = np.sqrt(sum(abs(c) ** 2 for c in state.values()))
    return {det: c / norm for det, c in state.items()}


@lru_cache(maxsize=128)
def _reference(nc, no, nv, lowering=0):
    nmo = nc + no + nv
    determinant = ((1 << (nc + no)) - 1) | (((1 << nc) - 1) << nmo)
    state = {determinant: 1.0}
    for _ in range(lowering):
        state = _normalize(_combine((1, _one_body(state, nmo + p, p)) for p in range(nmo)))
    return tuple(state.items())


def _tensor(state, p, q, projection, nmo):
    if projection == -1:
        return _one_body(state, nmo + p, q)
    if projection == 1:
        return _combine(((-1, _one_body(state, p, nmo + q)),))
    return _combine(
        (
            (1 / np.sqrt(2), _one_body(state, p, q)),
            (-1 / np.sqrt(2), _one_body(state, nmo + p, nmo + q)),
        )
    )


@lru_cache(maxsize=4096)
def _configuration(nc, no, nv, delta_s, p, q, rank=1):
    nmo, spin = nc + no + nv, no / 2
    reference = dict(_reference(nc, no, nv))
    if rank == 0:
        # NEST's CV(0) basis is -(a_alpha^dag a_alpha + a_beta^dag a_beta)/sqrt(2).
        state = _combine(
            (
                (-1 / np.sqrt(2), _one_body(reference, p, q)),
                (-1 / np.sqrt(2), _one_body(reference, nmo + p, nmo + q)),
            )
        )
    elif delta_s == 1:
        state = _tensor(reference, p, q, 1, nmo)
    else:
        if delta_s == 0:
            weights = (np.sqrt(spin / (spin + 1)), -1 / np.sqrt(spin + 1))
            projections = (0, 1)
        else:
            weights = (
                np.sqrt((2 * spin - 1) / (2 * spin + 1)),
                -np.sqrt((2 * spin - 1) / (spin * (2 * spin + 1))),
                1 / np.sqrt(spin * (2 * spin + 1)),
            )
            projections = (-1, 0, 1)
        state = _combine(
            (weight, _tensor(dict(_reference(nc, no, nv, lowering)), p, q, projection, nmo))
            for lowering, (weight, projection) in enumerate(zip(weights, projections))
        )
    if delta_s == -1:
        # OO is a redundant traceless representation. Normalizing each OO
        # configuration separately would change its metric and is incorrect.
        if q < nc and p >= nc + no:
            scale = 1.0
        elif q >= nc and p < nc + no:
            scale = np.sqrt((2 * spin - 1) / (2 * spin + 1))
        else:
            scale = np.sqrt(2 * spin / (2 * spin + 1))
        state = _combine(((scale, state),))
    else:
        state = _normalize(state)
    return tuple(state.items())


def wavefunction(tdobj, xy):
    """Expand one root in spin-adapted determinants, without renormalizing X."""
    nc, no, nv = map(len, _orbital_indices(tdobj))
    co, cv, oo, ov, cv0 = _split_amplitudes(xy[0], tdobj.deltaS, nc, no, nv)
    closed, open_, virtual = range(nc), range(nc, nc + no), range(nc + no, nc + no + nv)
    blocks = [(cv, closed, virtual, 1)]
    parts = []
    if tdobj.deltaS != 1:
        blocks.extend(((co, closed, open_, 1), (ov, open_, virtual, 1)))
        if tdobj.deltaS == 0:
            parts.append((oo, dict(_reference(nc, no, nv))))
            blocks.append((cv0, closed, virtual, 0))
        else:
            blocks.append((oo, open_, open_, 1))
    for block, sources, targets, rank in blocks:
        for i, q in enumerate(sources):
            for a, p in enumerate(targets):
                if abs(block[i, a]) > 1e-14:
                    config = _configuration(nc, no, nv, tdobj.deltaS, p, q, rank)
                    parts.append((block[i, a], dict(config)))
    return _combine(parts)


def _occupied(determinants, nmo, spin_offset):
    return np.asarray([[p for p in range(nmo) if det & (1 << (p + spin_offset))] for det in determinants], dtype=int)


def _overlaps(left_states, right_states, mo_overlap):
    """Batch determinant minors, then contract all requested pairs of roots."""
    left_dets = sorted(set().union(*left_states))
    right_dets = sorted(set().union(*right_states))
    if not left_dets or not right_dets:
        return np.zeros((len(left_states), len(right_states)))
    nl, nr = mo_overlap.shape
    occupied = [
        (_occupied(left_dets, nl, 0), _occupied(right_dets, nr, 0)),
        (_occupied(left_dets, nl, nl), _occupied(right_dets, nr, nr)),
    ]
    if any(l.shape[1] != r.shape[1] for l, r in occupied):
        return np.zeros((len(left_states), len(right_states)))
    left = np.asarray([[state.get(det, 0) for det in left_dets] for state in left_states])
    right = np.asarray([[state.get(det, 0) for det in right_dets] for state in right_states])
    result = np.zeros((len(left), len(right)), dtype=np.result_type(left, right, mo_overlap))
    # Limit the temporary integral-minor stack, rather than constructing a
    # four-dimensional array for every determinant pair at once.
    nelec = max(1, *(l.shape[1] for l, _ in occupied))
    batch = max(1, int(16e6 // (mo_overlap.itemsize * len(right_dets) * nelec**2)))
    for start in range(0, len(left_dets), batch):
        stop = min(start + batch, len(left_dets))
        minors = np.ones((stop - start, len(right_dets)), dtype=mo_overlap.dtype)
        for l, r in occupied:
            blocks = mo_overlap[l[start:stop, None, :, None], r[None, :, None, :]]
            minors *= np.linalg.det(blocks)
        result += left[:, start:stop].conj() @ minors @ right.T
    return result


def overlap_matrix(left_td, right_td, left_xy=None, right_xy=None):
    """Cross-geometry AWF overlaps; indices follow the supplied root lists."""
    left_xy = left_td.xy if left_xy is None else left_xy
    right_xy = right_td.xy if right_xy is None else right_xy
    left = [wavefunction(left_td, xy) for xy in left_xy]
    right = [wavefunction(right_td, xy) for xy in right_xy]
    ao = gto.intor_cross('int1e_ovlp', left_td.mol, right_td.mol)
    cl = left_td._scf.mo_coeff[:, np.concatenate(_orbital_indices(left_td))]
    cr = right_td._scf.mo_coeff[:, np.concatenate(_orbital_indices(right_td))]
    return _overlaps(left, right, cl.conj().T @ ao @ cr)


def _match_roots(reference, displaced, required, threshold):
    overlaps = overlap_matrix(reference, displaced)
    rows, columns = linear_sum_assignment(-np.abs(overlaps))
    assignment = dict(zip(rows, columns))
    matched = []
    for root in required:
        column = assignment.get(root)
        if column is None or abs(overlaps[root, column]) < threshold:
            raise RuntimeError('NTTDA state-tracking overlap is below %.3f for root %d' % (threshold, root + 1))
        if not displaced.converged[column]:
            raise RuntimeError('displaced NTTDA root did not converge')
        phase = 1 if overlaps[root, column] >= 0 else -1
        matched.append((phase * displaced.xy[column][0], 0))
    return matched


def finite_difference(nac, i, j, atoms, step):
    from nest.grad.nttda import _copy_td_settings, _displaced_reference
    from nest.nttda import NTTDA

    tdobj = nac.base
    coordinates = tdobj.mol.atom_coords()
    result = np.zeros((len(atoms), 3))
    for k, atom in enumerate(atoms):
        for xyz in range(3):
            displaced = []
            roots = []
            for sign in (-1, 1):
                coords = coordinates.copy()
                coords[atom, xyz] += sign * step
                mol = tdobj.mol.set_geom_(coords, unit='Bohr', inplace=False)
                mf = _displaced_reference(tdobj._scf, mol, nac.fixed_grid)
                mf.kernel(dm0=tdobj._scf.make_rdm1())
                if not mf.converged:
                    raise RuntimeError('displaced NTTDA reference did not converge')
                td = _copy_td_settings(tdobj, NTTDA(mf)).run()
                displaced.append(td)
                roots.append(_match_roots(tdobj, td, (i, j), nac.root_overlap_tol))
            overlaps = overlap_matrix(*displaced, left_xy=roots[0], right_xy=roots[1])
            result[k, xyz] = (overlaps[0, 1] - overlaps[1, 0]) / (4 * step)
    return result
