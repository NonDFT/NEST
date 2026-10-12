# Copyright 2026 The NEST Developers. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
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

The cofactor contraction is the derivative of det(M + t R) at t=0.
It remains valid when M is singular; det(M) * inv(M) must not be used
for such a target determinant. See Burton, J. Chem. Phys. 154, 144109
(2021), https://doi.org/10.1063/5.0045442, for nonorthogonal matrix elements.

Only real, nonrelativistic orbitals are supported, with a closed-shell RKS
bra and a two-SOMO triplet ROKS reference for deltaS=-1 NTTDA. The states
are used as supplied, without an extra orthogonalization or OO projection.
These are matrix elements of the determinant/CSF model wavefunctions,
not a TDDFT response derivation for independently optimized states.
"""

import warnings

import numpy as np
from pyscf import gto

from nest.nttda.nttda import _orbital_indices


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


def transition_properties(tdobj, ground, state=None, origin=None):
    """RKS -> singlet NTTDA overlaps, length-gauge dipoles and strengths.

    Args:
        tdobj: Computed deltaS=-1 NTTDA on a two-SOMO triplet ROKS reference.
            Frozen closed/virtual orbitals are supported; frozen closed
            orbitals remain occupied in the determinant matrix elements.
        ground: Converged closed-shell RKS at the same geometry and in the
            same AO basis/order. Its MOs may differ from the ROKS MOs.
        state: 1-based computed root or sequence of roots; None selects all.
        origin: Dipole origin in Bohr; None means (0, 0, 0).

    Returns:
        Dictionary of arrays, retaining a root axis even for one state:
        ``state`` (1-based), ``overlap`` = <RKS|I>, ``electronic_dipole`` =
        <RKS|sum_i (r_i-origin)|I>, ``dipole`` = electronic_dipole minus
        sum_A Z_A (R_A-origin) * overlap, ``excitation_energy`` =
        E_ROKS + omega_I - E_RKS (Hartree), and ``oscillator_strength`` =
        (2/3) excitation_energy * |dipole|^2. Dipoles have shape (n_roots,3)
        and units ea0. ``dipole`` is the NEGATIVE of the physical total
        electric dipole, preserving NEST's positive electronic-r convention.

    No roots are filtered: de-excitation roots may give negative energies
    and strengths. All input SCF calculations and selected roots should be
    converged. Root phases are arbitrary; compare norms/outer products.

    Including the nuclear term makes neutral-system results origin invariant
    even for nonzero overlap. For charge Q, moving the origin by a gives
    dipole' = dipole + Q*a*overlap; a warning is emitted when Q != 0 and
    max|overlap| > 1e-8. The states are not orthogonalized, so strengths are
    an approximate use of the usual oscillator-strength expression.
    """
    mf, mol = tdobj._scf, tdobj._scf.mol
    cg, ct = np.asarray(ground.mo_coeff), np.asarray(mf.mo_coeff)
    ground_occ, ref_occ = np.asarray(ground.mo_occ), np.asarray(mf.mo_occ)
    if (cg.ndim != 2 or ct.ndim != 2 or ground_occ.ndim != 1 or ref_occ.ndim != 1
            or not np.all(np.isin(ground_occ, [0, 2])) or not np.all(np.isin(ref_occ, [0, 1, 2]))):
        raise ValueError('Real restricted RKS/ROKS orbitals with integer occupations are required')
    cs, os, vs = _orbital_indices(tdobj)
    reference_cs = np.flatnonzero(ref_occ == 2)
    if tdobj.deltaS != -1 or len(os) != 2 or mol.spin != 2 or ground.mol.spin != 0:
        raise ValueError('Requires a closed-shell ground state and deltaS=-1 NTTDA on a triplet ROKS reference')
    if (ground.mol.nelectron != mol.nelectron or np.count_nonzero(ground_occ == 2) != len(reference_cs) + 1):
        raise ValueError('RKS and NTTDA must have the same number of electrons')
    if (ground.mol.natm != mol.natm or ground.mol.cart != mol.cart
            or [ground.mol.atom_symbol(i) for i in range(mol.natm)] != [mol.atom_symbol(i) for i in range(mol.natm)]
            or not np.allclose(ground.mol.atom_coords(), mol.atom_coords(), atol=1e-10, rtol=0)
            or not gto.same_basis_set(ground.mol, mol)):
        raise ValueError('RKS and NTTDA must use the same geometry and AO basis/order')
    if getattr(ground, 'with_x2c', None) is not None or getattr(mf, 'with_x2c', None) is not None:
        raise NotImplementedError('X2C transition operators are not implemented here')
    states = np.arange(1, len(tdobj.xy) + 1) if state is None else np.atleast_1d(state)
    if states.ndim != 1 or not len(states) or np.any(states != states.astype(int)):
        raise ValueError('state must contain integer computed-root indices (1-based)')
    states = states.astype(int)
    if np.any(states < 1) or np.any(states > len(tdobj.xy)):
        raise ValueError('state must index computed roots (1-based)')
    amplitudes = np.array([np.asarray(tdobj.xy[i - 1][0]).ravel() for i in states])
    if any(np.iscomplexobj(a) for a in (cg, ct, amplitudes)):
        raise NotImplementedError('Complex orbitals/amplitudes are not implemented here')

    cg = cg[:, ground_occ == 2]
    origin = np.zeros(3) if origin is None else np.asarray(origin, dtype=float)
    cross_s = cg.T @ ground.get_ovlp() @ ct
    with mol.with_common_orig(origin):
        dip_ao = mol.intor_symmetric('int1e_r', comp=3)
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
    # conditioned overlap, without assuming that a particular SOMO is core.
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
    electronic = amplitudes @ basis_dipole
    nuclear = mol.atom_charges() @ (mol.atom_coords() - origin)
    dipole = electronic - overlap[:, None] * nuclear
    energy = mf.e_tot + np.asarray(tdobj.e)[states - 1] - ground.e_tot
    strength = (2.0 / 3.0) * energy * np.einsum('nx,nx->n', dipole, dipole)
    if mol.charge != 0 and np.max(np.abs(overlap)) > 1e-8:
        warnings.warn(
            f'Nonorthogonal RKS/NTTDA states in a charged system (charge={mol.charge}, '
            f'max|overlap|={np.max(np.abs(overlap)):.3g}): transition dipoles and '
            'oscillator strengths depend on the chosen origin even with the nuclear term.',
            UserWarning, stacklevel=2,
        )
    return dict(state=states, overlap=overlap, electronic_dipole=electronic, dipole=dipole,
                excitation_energy=energy, oscillator_strength=strength)
