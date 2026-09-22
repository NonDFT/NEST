"""Analytic gradient for current NTTDA ``deltaS=0``.

The public ``grad_elec`` function exposes the complete scientific data flow.
Same-spin amplitudes, Fock projections and response coefficients live here.
Reference response, J/K derivatives, XC quadrature and the adjoint are shared.
"""

from dataclasses import dataclass

import numpy as np

from pyscf import lib
from nest.nttda import nttda as nttda_mod
from nest.nttda.nttda import gen_rohf_response_sc

from .common import (
    assemble_gradient,
    orbital_spaces,
    pair_density,
    FockProjection,
    _fock_response_q,
    ResponseTerm,
    _apply_reference_responses,
    _apply_hfx_responses,
)


# Orbital spaces and native amplitudes


@dataclass(frozen=True)
class SameSpinAmplitudes:
    """Five amplitude blocks used by ``NTTDA(deltaS=0)``."""

    co: np.ndarray
    cv: np.ndarray
    oo: float
    ov: np.ndarray
    cv0: np.ndarray


def same_spin_slices(spaces):
    """Return canonical slices for ``CO/CV/OO/OV/CV0`` amplitudes."""
    nc = len(spaces.closed)
    no = len(spaces.open)
    nv = len(spaces.virtual)
    nco = nc * no
    ncv = nc * nv
    nov = no * nv
    i1 = nco
    i2 = i1 + ncv
    i3 = i2 + 1
    i4 = i3 + nov
    return {
        "CO": slice(0, i1),
        "CV": slice(i1, i2),
        "OO": slice(i2, i3),
        "OV": slice(i3, i4),
        "CV0": slice(i4, i4 + ncv),
    }


def split_same_spin(tdobj, xy):
    """Split one packed ``deltaS=0`` vector into its five native blocks."""
    spaces = orbital_spaces(tdobj)
    if spaces.spin < 0.5:
        raise ValueError("NTTDA deltaS=0 requires at least one open orbital")
    vector = xy[0] if isinstance(xy, (tuple, list)) else xy
    vector = np.asarray(vector).reshape(-1)
    slices = same_spin_slices(spaces)
    expected = slices["CV0"].stop
    if vector.size != expected:
        raise ValueError(
            "deltaS=0 amplitude has size %d; expected %d" %
            (vector.size, expected)
        )
    nc = len(spaces.closed)
    no = len(spaces.open)
    nv = len(spaces.virtual)
    return spaces, SameSpinAmplitudes(
        co=vector[slices["CO"]].reshape(nc, no),
        cv=vector[slices["CV"]].reshape(nc, nv),
        oo=float(vector[slices["OO"]][0]),
        ov=vector[slices["OV"]].reshape(no, nv),
        cv0=vector[slices["CV0"]].reshape(nc, nv),
    )


def same_spin_transition_densities(tdobj, xy):
    """Return directed AO transition densities for the four response blocks."""
    spaces, amp = split_same_spin(tdobj, xy)
    return spaces, amp, {
        "CO": pair_density(spaces.c_open, amp.co.T, spaces.c_closed),
        "CV": pair_density(spaces.c_virtual, amp.cv.T, spaces.c_closed),
        "OV": pair_density(spaces.c_virtual, amp.ov.T, spaces.c_open),
        "CV0": pair_density(spaces.c_virtual, amp.cv0.T, spaces.c_closed),
    }


# Explicit F0/Fz ledger


def fock0_fockz(tdobj, max_memory=None):
    """Build exactly the ``F0`` and ``Fz`` matrices used by ``gen_vind_sc``."""
    mf = tdobj._scf
    if max_memory is None:
        max_memory = tdobj.max_memory
    _response, fockz = gen_rohf_response_sc(
        mf,
        mo_coeff=mf.mo_coeff,
        mo_occ=mf.mo_occ,
        hermi=0,
        max_memory=max_memory,
    )
    fock0 = nttda_mod._reference_fock0(mf, tdobj.nobeta)
    return fock0, fockz


def same_spin_fock_projections(tdobj, xy):
    """Return the complete five-block explicit-Fock ledger."""
    spaces, x = split_same_spin(tdobj, xy)
    spin = spaces.spin
    c = spaces.c_closed
    o = spaces.c_open
    v = spaces.c_virtual
    a = np.sqrt((spin + 1.0) / (2.0 * spin))
    b = np.sqrt(2.0 * (spin + 1.0) / spin)
    d = np.sqrt((spin + 1.0) / spin)
    h = np.sqrt(0.5)

    terms = []

    def indices(orbitals):
        if orbitals is c:
            return spaces.closed
        if orbitals is o:
            return spaces.open
        if orbitals is v:
            return spaces.virtual
        raise ValueError("Fock projection uses an unknown orbital space")

    def add(name, left, coefficient, right, f0, fz):
        coefficient = np.asarray(coefficient)
        if coefficient.size:
            terms.append(FockProjection(
                name,
                indices(left), left, coefficient,
                indices(right), right,
                float(f0), float(fz),
            ))

    # CO row/column and its couplings.
    add("co-oo", o, x.co.T @ x.co, o, 1.0, -1.0)
    add("co-cc", c, -x.co @ x.co.T, c, 1.0, -1.0)
    add("co-cv", o, 2.0 * a * (x.co.T @ x.cv), v, 1.0, -1.0)
    add("co-oo1", o, -2.0 * x.oo * x.co.T, c, 1.0, -1.0)
    add("co-cv0", o, 2.0 * h * (x.co.T @ x.cv0), v, 1.0, -1.0)

    # CV block and its OO/OV/CV0 couplings.
    add("cv-vv", v, x.cv.T @ x.cv, v, 1.0, -1.0 / spin)
    add("cv-cc", c, -x.cv @ x.cv.T, c, 1.0, 1.0 / spin)
    add("cv-oo1", v, 2.0 * b * x.oo * x.cv.T, c, 0.0, 1.0)
    add("cv-ov", o, -2.0 * a * (x.ov @ x.cv.T), c, 1.0, 1.0)
    add(
        "cv-cv0-vv", v,
        -d * (x.cv.T @ x.cv0 + x.cv0.T @ x.cv), v, 0.0, 1.0,
    )
    add(
        "cv-cv0-cc", c,
        d * (x.cv0 @ x.cv.T + x.cv @ x.cv0.T), c, 0.0, 1.0,
    )

    # OV and CV0 diagonal/coupling terms.
    add("ov-vv", v, x.ov.T @ x.ov, v, 1.0, 1.0)
    add("ov-oo", o, -x.ov @ x.ov.T, o, 1.0, 1.0)
    add("ov-oo1", v, 2.0 * x.oo * x.ov.T, o, 1.0, 1.0)
    add("ov-cv0", c, 2.0 * h * (x.cv0 @ x.ov.T), o, 1.0, 1.0)
    add("cv0-vv", v, x.cv0.T @ x.cv0, v, 1.0, 0.0)
    add("cv0-cc", c, -x.cv0 @ x.cv0.T, c, 1.0, 0.0)
    add("cv0-oo1", v, -2.0 * np.sqrt(2.0) * x.oo * x.cv0.T, c, 1.0, 0.0)
    return tuple(terms)


def same_spin_fock_probes(tdobj, xy):
    """Return AO probes ``(P0, Pz)`` generated from the Fock ledger."""
    nao = tdobj.mol.nao_nr()
    p0 = np.zeros((nao, nao))
    pz = np.zeros_like(p0)
    for term in same_spin_fock_projections(tdobj, xy):
        density = term.density()
        p0 += term.weight_f0 * density
        pz += term.weight_fz * density
    return p0, pz


def same_spin_fock_q(tdobj, xy, max_memory=None):
    """Return the explicit-Fock contribution to ``(Q_alpha,Q_beta)``.

    For HF the complete ``Fz`` response is exactly represented by the
    spin-resolved probes.  DFT callers add the independent ``Fz`` and
    ``nobeta`` response ledgers after this common ``F0`` contribution.
    """
    mf = tdobj._scf
    mo = np.asarray(mf.mo_coeff)
    nmo = mo.shape[1]
    fock0, fockz = fock0_fockz(tdobj, max_memory=max_memory)
    fock0_mo = mo.conj().T @ fock0 @ mo
    fockz_mo = mo.conj().T @ fockz @ mo
    q_alpha = np.zeros((nmo, nmo))
    q_beta = np.zeros_like(q_alpha)

    for term in same_spin_fock_projections(tdobj, xy):
        left = term.left_indices
        right = term.right_indices
        coeff = term.coefficient

        def project(target, operator, scale):
            if scale == 0.0:
                return
            target[:, left] += scale * operator[:, right] @ coeff.T
            target[:, right] += scale * operator[:, left] @ coeff

        project(q_alpha, fock0_mo, 0.5 * term.weight_f0)
        project(q_beta, fock0_mo, 0.5 * term.weight_f0)
        if mf._numint._xc_type(mf.xc) == "HF":
            project(q_alpha, fockz_mo, 0.5 * term.weight_fz)
            project(q_beta, fockz_mo, 0.5 * term.weight_fz)
        else:
            project(q_alpha, fockz_mo, term.weight_fz)

    p0, pz = same_spin_fock_probes(tdobj, xy)
    p_alpha = 0.5 * p0
    p_beta = 0.5 * p0
    if mf._numint._xc_type(mf.xc) == "HF":
        p_alpha = p_alpha + 0.5 * pz
        p_beta = p_beta - 0.5 * pz
    response_alpha, response_beta = _fock_response_q(
        tdobj, p_alpha, p_beta,
    )
    q_alpha += response_alpha
    q_beta += response_beta
    return q_alpha, q_beta


def same_spin_fock_scalar(tdobj, xy, max_memory=None):
    """Evaluate the complete explicit-Fock part of ``X.T A_sc X``."""
    fock0, fockz = fock0_fockz(tdobj, max_memory=max_memory)
    return same_spin_fock_projection_scalar(tdobj, xy, fock0, fockz)


def same_spin_fock_projection_scalar(tdobj, xy, fock0, fockz):
    """Evaluate the Fock ledger for caller-supplied frozen operators."""
    p0, pz = same_spin_fock_probes(tdobj, xy)
    return float(
        lib.einsum("pq,pq->", p0, fock0)
        + lib.einsum("pq,pq->", pz, fockz)
    )


# vref0/vref1 response ledger


def same_spin_response_terms(spin):
    """Directed coefficients transcribed from ``gen_rohf_response_sc``."""
    a = np.sqrt((spin + 1.0) / (2.0 * spin))
    h = np.sqrt(0.5)
    r2 = np.sqrt(2.0)
    return (
        ResponseTerm("CO", "CO", 1.0, -1.0),
        ResponseTerm("CO", "CV", a, 0.0),
        ResponseTerm("CO", "OV", 0.0, 1.0),
        ResponseTerm("CO", "CV0", h, -r2),
        ResponseTerm("CV", "CO", a, 0.0),
        ResponseTerm("CV", "CV", 1.0, 0.0),
        ResponseTerm("CV", "OV", a, 0.0),
        ResponseTerm("OV", "CO", 0.0, 1.0),
        ResponseTerm("OV", "CV", a, 0.0),
        ResponseTerm("OV", "OV", 1.0, -1.0),
        ResponseTerm("OV", "CV0", -h, r2),
        ResponseTerm("CV0", "CO", h, -r2),
        ResponseTerm("CV0", "OV", -h, r2),
        ResponseTerm("CV0", "CV0", 1.0, -2.0),
    )


def _derivative_potentials(spaces, densities, vref0, vref1):
    potentials = {label: np.zeros_like(dm) for label, dm in densities.items()}
    for term in same_spin_response_terms(spaces.spin):
        if term.vref0:
            potentials[term.target] += term.vref0 * vref0[term.source]
            potentials[term.source] += term.vref0 * vref0[term.target]
        if term.vref1:
            potentials[term.target] += term.vref1 * vref1[term.source]
            potentials[term.source] += term.vref1 * vref1[term.target]
    return potentials


def same_spin_response_scalar(tdobj, xy, max_memory=None):
    """Evaluate all current-NTTDA response terms in ``X.T A_sc X``."""
    spaces, _amp, densities = same_spin_transition_densities(tdobj, xy)
    vref0, vref1 = _apply_reference_responses(
        tdobj, densities, max_memory=max_memory,
    )
    value = 0.0
    for term in same_spin_response_terms(spaces.spin):
        target = densities[term.target]
        if term.vref0:
            value += term.vref0 * lib.einsum(
                "pq,pq->", target, vref0[term.source],
            )
        if term.vref1:
            value += term.vref1 * lib.einsum(
                "pq,pq->", target, vref1[term.source],
            )
    return float(value)


def same_spin_response_derivative_potentials(tdobj, xy, max_memory=None):
    """AO potentials obtained by varying both sides of the response scalar."""
    spaces, _amp, densities = same_spin_transition_densities(tdobj, xy)
    vref0, vref1 = _apply_reference_responses(
        tdobj, densities, max_memory=max_memory,
    )
    return _derivative_potentials(spaces, densities, vref0, vref1)


def same_spin_response_projection_q(tdobj, xy, max_memory=None):
    """MO derivative from transition-density factors at frozen kernels."""
    spaces, amp, _densities = same_spin_transition_densities(tdobj, xy)
    potentials = same_spin_response_derivative_potentials(
        tdobj, xy, max_memory=max_memory,
    )
    mo = np.asarray(tdobj._scf.mo_coeff)
    q_alpha = np.zeros((mo.shape[1], mo.shape[1]))
    q_beta = np.zeros_like(q_alpha)
    block_data = {
        "CO": (spaces.open, spaces.closed, amp.co.T),
        "CV": (spaces.virtual, spaces.closed, amp.cv.T),
        "OV": (spaces.virtual, spaces.open, amp.ov.T),
        "CV0": (spaces.virtual, spaces.closed, amp.cv0.T),
    }
    for label, (target, source, coefficient) in block_data.items():
        potential_mo = mo.conj().T @ potentials[label] @ mo
        q_beta[:, target] += potential_mo[:, source] @ coefficient.T
        q_alpha[:, source] += potential_mo[target, :].T @ coefficient
    return q_alpha, q_beta


def same_spin_hfx_projection_q(tdobj, xy):
    """Transition-factor derivative of only the hybrid/RSH response scalar."""
    spaces, amp, densities = same_spin_transition_densities(tdobj, xy)
    vref0, vref1 = _apply_hfx_responses(tdobj, densities)
    potentials = _derivative_potentials(
        spaces, densities, vref0, vref1,
    )
    mo = np.asarray(tdobj._scf.mo_coeff)
    q_alpha = np.zeros((mo.shape[1], mo.shape[1]))
    q_beta = np.zeros_like(q_alpha)
    block_data = {
        "CO": (spaces.open, spaces.closed, amp.co.T),
        "CV": (spaces.virtual, spaces.closed, amp.cv.T),
        "OV": (spaces.virtual, spaces.open, amp.ov.T),
        "CV0": (spaces.virtual, spaces.closed, amp.cv0.T),
    }
    for label, (target, source, coefficient) in block_data.items():
        potential_mo = mo.conj().T @ potentials[label] @ mo
        q_beta[:, target] += potential_mo[:, source] @ coefficient.T
        q_alpha[:, source] += potential_mo[target, :].T @ coefficient
    return q_alpha, q_beta


# Scalar closure diagnostics (private to this channel)

def same_spin_action_scalar(tdobj, xy):
    """Evaluate ``X.T gen_vind_sc(X)`` using the production NTTDA action."""
    vector = xy[0] if isinstance(xy, (tuple, list)) else xy
    vector = np.asarray(vector).reshape(-1)
    vind, _hdiag = tdobj.gen_vind_sc()
    action = vind(vector.reshape(1, -1))[0]
    return float(np.vdot(vector, action).real)


def same_spin_ledger_scalar(tdobj, xy, max_memory=None, return_parts=False):
    """Evaluate the independent ``F0/Fz + vref0/vref1`` scalar ledger."""
    fock = same_spin_fock_scalar(tdobj, xy, max_memory=max_memory)
    response = same_spin_response_scalar(tdobj, xy, max_memory=max_memory)
    total = fock + response
    if return_parts:
        return {"fock": fock, "response": response, "total": total}
    return total

# Channel assembly

def grad_elec(
        gradient_driver, tdobj, xy, atmlst=None, tolerance=1e-12,
        max_cycle=None):
    """Build the analytic excitation gradient for deltaS=0."""
    if tdobj.deltaS != 0:
        raise ValueError("deltaS=0 gradient received a different spin channel")
    spaces, amplitudes, densities = same_spin_transition_densities(tdobj, xy)
    blocks = {
        "CO": (spaces.open, spaces.closed, amplitudes.co.T),
        "CV": (spaces.virtual, spaces.closed, amplitudes.cv.T),
        "OV": (spaces.virtual, spaces.open, amplitudes.ov.T),
        "CV0": (spaces.virtual, spaces.closed, amplitudes.cv0.T),
    }
    response_terms = same_spin_response_terms(spaces.spin)
    channel_data = (spaces, amplitudes, densities, blocks, response_terms)
    p0, pz = same_spin_fock_probes(tdobj, xy)
    return assemble_gradient(
        gradient_driver, tdobj, channel_data, (p0, pz),
        same_spin_fock_q(tdobj, xy),
        same_spin_hfx_projection_q(tdobj, xy),
        atmlst=atmlst, tolerance=tolerance, max_cycle=max_cycle,
    )
