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

"""Independent-orbital RKS -> singlet NTTDA transition properties.

The RKS determinant and the NTTDA configuration expansion need not be
orthogonal. For each spin, M = C_RKS,occ.T @ S_AO @ C_det,occ and
R = C_RKS,occ.T @ r_AO @ C_det,occ give

    overlap = det(M_alpha) det(M_beta)
    <RKS|sum_i r_i|det> = det(M_beta) sum_ij cof(M_alpha)_ij R_alpha,ij
                       + det(M_alpha) sum_ij cof(M_beta)_ij R_beta,ij.

Ref: JCP, 2021, 154, 144109.
"""

import numpy as np
from pyscf.lib import logger

from nest.nttda.nttda import _orbital_indices, _dipole_integrals


def _apply_e(state, p, q):
    """Apply a_p^dagger a_q, with alpha spin orbitals preceding beta."""
    result = {}
    for det, coefficient in state.items():
        if not (det >> q) & 1:
            continue
        sign = (-1) ** (det & ((1 << q) - 1)).bit_count()
        remaining = det ^ (1 << q)
        if (remaining >> p) & 1:
            continue
        sign *= (-1) ** (remaining & ((1 << p) - 1)).bit_count()
        out = remaining | (1 << p)
        result[out] = result.get(out, 0.0) + sign * coefficient
    return result


def _add_states(*terms):
    result = {}
    for factor, state in terms:
        for det, coefficient in state.items():
            result[det] = result.get(det, 0.0) + factor * coefficient
    return {det: coefficient for det, coefficient in result.items() if coefficient != 0}


def _tensor(state, b, j, component, nmo):
    # Spherical-tensor phases match NTTDA's CO/CV/OO/OV amplitudes.
    if component == -1:
        return _apply_e(state, b + nmo, j)
    if component == 1:
        return _add_states((-1, _apply_e(state, b, j + nmo)))
    return _add_states((1 / np.sqrt(2), _apply_e(state, b, j)),
                       (-1 / np.sqrt(2), _apply_e(state, b + nmo, j + nmo)))


def _singlet_csfs(cs, os, vs, nmo, reference_cs=None):
    """Yield S=1 -> S=0 CSFs in stored [source(C,O), target(O,V)] order."""
    # Python ints (not numpy int64) are needed for arbitrarily many orbitals.
    cs, os, vs = list(map(int, cs)), list(map(int, os)), list(map(int, vs))
    reference_cs = cs if reference_cs is None else list(map(int, reference_cs))
    # Frozen closed orbitals remain doubly occupied in every determinant;
    # only the excitation labels below use the active C/O/V spaces.
    ref_det = sum(1 << p for p in reference_cs + os) + sum(1 << (p + nmo) for p in reference_cs)
    ref = {ref_det: 1.0}
    lowered = _add_states(*[(1, _tensor(ref, w, w, -1, nmo)) for w in os])
    double_flip = _tensor(_tensor(ref, os[0], os[0], -1, nmo), os[1], os[1], -1, nmo)

    for j in cs + os:
        for b in os + vs:
            bare = _tensor(ref, b, j, -1, nmo)
            if j in cs and b in vs:
                # CV: T-/sqrt(3) - T0 sum_w T-ww/sqrt(6)
                #     + T+ T-o1o1 T-o2o2/sqrt(3).
                yield _add_states(
                    (1 / np.sqrt(3), bare),
                    (-1 / np.sqrt(6), _tensor(lowered, b, j, 0, nmo)),
                    (1 / np.sqrt(3), _tensor(double_flip, b, j, 1, nmo)),
                )
            elif j in cs and b in os:
                w = next(w for w in os if w != b)
                yield _add_states(
                    (1 / np.sqrt(2), bare),
                    (1 / np.sqrt(2), _apply_e(_tensor(ref, w, w, -1, nmo), b + nmo, j + nmo)),
                )
            elif j in os and b in vs:
                w = next(w for w in os if w != j)
                yield _add_states(
                    (1 / np.sqrt(2), bare),
                    (-1 / np.sqrt(2), _apply_e(_tensor(ref, w, w, -1, nmo), b, j)),
                )
            else:
                # OO: keep T-bj as in the stored amplitudes. Physical roots
                # already exclude the redundant trace direction in NTTDA.
                yield bare


def _svd_element(matrix, operator):
    """det(M) and d det(M+tR)/dt, including rank-deficient M (real)."""
    u, singular, vt = np.linalg.svd(matrix)
    phase = np.linalg.det(u) * np.linalg.det(vt)
    # Products excluding each singular value, with no division by zero and
    # no rank threshold: a single zero can still give a nonzero matrix element.
    left = np.concatenate(([1.0], np.cumprod(singular[:-1])))
    right = np.concatenate((np.cumprod(singular[:0:-1])[::-1], [1.0]))
    diagonal = np.einsum('pi,xpq,iq->xi', u, operator, vt)
    return phase * np.prod(singular), phase * (diagonal @ (left * right))


def _spin_elements(cross_s, cross_r, occupations, filled):
    """One spin's occupied determinants; reuse one auxiliary overlap matrix.

    F fills all closed orbitals plus one SOMO. Each S=1 CSF determinant
    differs from this closed-shell F by at most two column replacements
    in total. Let P=M0^-1 L, F_r=M0^-1 R0 and Q=M0^-1 R-F_r P.
    For replaced slots H and incoming orbitals A, K=P[H,A], and

      det(M) = phase det(M0) det(K)
      d det(M+tR)/dt = phase det(M0) [tr(F_r) det(K) + cof(K):Q[H,A]].

    K is at most 2x2, even if a target occupied overlap is singular. Only
    the auxiliary M0 is solved against. If M0 is ill-conditioned we instead
    evaluate each occupied matrix by SVD, at O(n_occ^3) per spin determinant.
    """
    m0 = cross_s[:, filled]
    # Absolute singular values matter: cond(M0)=1 can hide a very small M0.
    # The orbitals are orthonormal within each MO set, so singular values <=1.
    fast = np.linalg.svd(m0, compute_uv=False)[-1] > 1e-2
    if fast:
        det0 = np.linalg.det(m0)
        p = np.linalg.solve(m0, cross_s)
        f0 = np.array([np.linalg.solve(m0, r[:, filled]) for r in cross_r])
        trace = np.trace(f0, axis1=1, axis2=2)
        q = np.array([np.linalg.solve(m0, r) for r in cross_r]) - f0 @ p

    result = {}
    filled_set = set(filled)
    for occupied in occupations:
        if not fast:
            result[occupied] = _svd_element(cross_s[:, occupied], cross_r[:, :, occupied])
            continue
        occupied_set = set(occupied)
        holes = [i for i, orbital in enumerate(filled) if orbital not in occupied_set]
        particles = [orbital for orbital in occupied if orbital not in filled_set]
        reordered = list(filled)
        for i, orbital in zip(holes, particles):
            reordered[i] = orbital
        # The update uses the old column slots; restore sorted orbital order.
        # Unchanged columns remain sorted. Count only pairs involving one
        # of the <=2 replaced slots, in O(n_occ) rather than O(n_occ^2).
        inversions = sum(reordered[i] > reordered[j] for i in holes for j in range(i + 1, len(filled)))
        inversions += sum(reordered[i] > reordered[j] for j in holes for i in range(j) if i not in holes)
        prefactor = (-1) ** inversions * det0
        k = p[np.ix_(holes, particles)]
        dq = q[:, holes][:, :, particles]
        if len(holes) == 0:
            determinant, derivative = 1.0, np.zeros(len(cross_r))
        elif len(holes) == 1:
            determinant, derivative = k[0, 0], dq[:, 0, 0]
        else:
            determinant = k[0, 0] * k[1, 1] - k[0, 1] * k[1, 0]
            derivative = (k[1, 1] * dq[:, 0, 0] + k[0, 0] * dq[:, 1, 1]
                          - k[1, 0] * dq[:, 0, 1] - k[0, 1] * dq[:, 1, 0])
        result[occupied] = (prefactor * determinant, prefactor * (trace * determinant + derivative))
    return result


def transition_properties(tdobj, ground, state=None, origin=None, gauge='length', picture_change=True):
    """Transition properties from a closed-shell RKS ground state to singlet NTTDA.

    Use deltaS=-1 NTTDA on a triplet ROKS reference. Both SCF objects must
    have the same geometry, AO basis and relativistic settings; their MOs
    may differ. Only real orbitals and amplitudes are supported.
    ``state`` selects 1-based roots; None selects all computed roots.

    Returns a dictionary with ``state``, ``overlap``, ``excitation_energy``
    (E_ROKS + omega - E_RKS, Hartree), and ``oscillator_strength``.
    Length gauge adds ``electronic_dipole`` and ``dipole`` (including the
    nuclear term, positive electronic-r convention). Velocity gauge adds
    only ``velocity_dipole`` = i<v>. Moments have shape (n_states, 3), in a.u.

    The dipole origin defaults to the center of mass; an explicit ``origin``
    is in Bohr. sfX2C1e operators include picture change by default.
    For neutral systems, the nuclear term makes total dipoles origin independent.
    States are not orthogonalized; the two gauges need not agree.
    """
    mf, mol = tdobj._scf, tdobj._scf.mol
    cg, ct = np.asarray(ground.mo_coeff), np.asarray(mf.mo_coeff)
    ground_occ, ref_occ = np.asarray(ground.mo_occ), np.asarray(mf.mo_occ)
    cs, os, vs = _orbital_indices(tdobj)
    reference_cs = np.flatnonzero(ref_occ == 2)
    if tdobj.deltaS != -1 or len(os) != 2 or mol.spin != 2 or ground.mol.spin != 0:
        raise ValueError('Requires a closed-shell RKS ground state and deltaS=-1 NTTDA on a triplet ROKS reference')
    if ground.mol.nelectron != mol.nelectron:
        raise ValueError('RKS and NTTDA must have the same number of electrons')
    gx = getattr(ground, 'with_x2c', None) or None
    tx = getattr(mf, 'with_x2c', None) or None
    if ((gx is None) != (tx is None)
            or gx is not None and any(getattr(gx, k) != getattr(tx, k) for k in ('approx', 'basis', 'xuncontract'))):
        raise ValueError('RKS and ROKS must use matching nonrelativistic or sfX2C1e settings')
    states = np.arange(1, len(tdobj.xy) + 1) if state is None else np.atleast_1d(state).astype(int)
    if np.any(states < 1) or np.any(states > len(tdobj.xy)):
        raise ValueError('state must index computed roots (1-based)')
    amplitudes = np.array([tdobj.xy[i - 1][0].ravel() for i in states])
    if any(np.iscomplexobj(a) for a in (cg, ct, amplitudes)):
        raise NotImplementedError('Complex orbitals/amplitudes are not implemented here')

    cg = cg[:, ground_occ == 2]
    if origin is None:
        masses = mol.atom_mass_list()
        origin = masses @ mol.atom_coords() / masses.sum()
    cross_s = cg.T @ ground.get_ovlp() @ ct
    with mol.with_common_orig(origin):
        dip_ao = _dipole_integrals(mf, gauge, picture_change=picture_change)
    cross_r = np.einsum('up,xuv,vq->xpq', cg, dip_ao, ct, optimize=True)

    basis = list(_singlet_csfs(cs, os, vs, ct.shape[1], reference_cs))
    determinants = set().union(*basis)
    occupations = {}
    for det in determinants:
        alpha = tuple(p for p in range(ct.shape[1]) if (det >> p) & 1)
        beta = tuple(p for p in range(ct.shape[1]) if (det >> (p + ct.shape[1])) & 1)
        occupations[det] = alpha, beta
    spin_occupations = {occupied for spins in occupations.values() for occupied in spins}
    # Either SOMO can define the auxiliary closed shell. Choose the better
    # conditioned overlap matrix without assuming which SOMO is the core.
    candidates = [sorted([*reference_cs, orbital]) for orbital in os]
    filled = max(candidates, key=lambda occupied: np.linalg.svd(cross_s[:, occupied], compute_uv=False)[-1])
    elements = _spin_elements(cross_s, cross_r, spin_occupations, filled)
    basis_overlap = np.zeros(len(basis))
    basis_dipole = np.zeros((len(basis), 3))
    for i, csf in enumerate(basis):
        for det, coefficient in csf.items():
            alpha, beta = occupations[det]
            sa, ra = elements[alpha]
            sb, rb = elements[beta]
            basis_overlap[i] += coefficient * sa * sb
            basis_dipole[i] += coefficient * (sb * ra + sa * rb)
    overlap = amplitudes @ basis_overlap
    moments = amplitudes @ basis_dipole
    energy = mf.e_tot + np.asarray(tdobj.e)[states - 1] - ground.e_tot
    result = dict(state=states, overlap=overlap, excitation_energy=energy)
    if gauge == 'length':
        nuclear = mol.atom_charges() @ (mol.atom_coords() - origin)
        dipole = moments - overlap[:, None] * nuclear
        result['electronic_dipole'] = moments
        result['dipole'] = dipole
        result['oscillator_strength'] = (2.0 / 3.0) * energy * np.einsum('nx,nx->n', dipole, dipole)
        if mol.charge != 0 and np.max(np.abs(overlap)) > 1e-8:
            logger.warn(tdobj, 'Nonorthogonal RKS/NTTDA states in a charged system: '
                        'length-gauge strengths depend on the chosen origin.')
    else:
        if np.any(energy == 0):
            raise ValueError('Velocity-gauge oscillator strengths require nonzero transition energies')
        result['velocity_dipole'] = moments
        result['oscillator_strength'] = (2.0 / 3.0) * np.einsum('nx,nx->n', moments, moments) / energy
    return result
