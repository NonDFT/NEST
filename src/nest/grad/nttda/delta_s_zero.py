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
    fock_probes,
    fock_projection_q,
    response_projection_q,
    ResponseTerm,
    _apply_reference_responses,
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


def same_spin_block_data(spaces, amplitudes):
    """MO index/factor map for variations of same-spin transition densities."""
    return {
        "CO": (spaces.open, spaces.closed, amplitudes.co.T),
        "CV": (spaces.virtual, spaces.closed, amplitudes.cv.T),
        "OV": (spaces.virtual, spaces.open, amplitudes.ov.T),
        "CV0": (spaces.virtual, spaces.closed, amplitudes.cv0.T),
    }


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
    """AO probes of the channel's explicit Fock scalar."""
    return fock_probes(tdobj, same_spin_fock_projections(tdobj, xy))


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
    blocks = same_spin_block_data(spaces, amplitudes)
    response_terms = same_spin_response_terms(spaces.spin)
    channel_data = (spaces, amplitudes, densities, blocks, response_terms)
    projections = same_spin_fock_projections(tdobj, xy)
    probes = fock_probes(tdobj, projections)
    return assemble_gradient(
        gradient_driver, tdobj, channel_data, probes,
        fock_projection_q(tdobj, projections, fock0_fockz(tdobj), probes),
        response_projection_q(tdobj, channel_data, hfx_only=True),
        atmlst=atmlst, tolerance=tolerance, max_cycle=max_cycle,
    )
