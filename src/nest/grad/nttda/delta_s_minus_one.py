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

"""Analytic gradient for current NTTDA ``deltaS=-1``.

This module owns the spin-lowering amplitudes, Fock projections and response
coefficients. Reference response and AO derivatives are shared between channels.
"""

from dataclasses import dataclass

import numpy as np

from pyscf import lib
from nest.nttda.nttda import gen_rohf_response_sfd

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
class SpinLoweringAmplitudes:
    """Four native blocks used by ``NTTDA(deltaS=-1)``."""

    co: np.ndarray
    cv: np.ndarray
    oo: np.ndarray
    ov: np.ndarray


def split_spin_lowering(tdobj, xy):
    """Split a lowering-channel amplitude into ``CO/CV/OO/OV`` blocks."""
    spaces = orbital_spaces(tdobj)
    if spaces.spin < 1.0:
        raise ValueError("NTTDA deltaS=-1 requires reference spin Si >= 1")
    vector = xy[0] if isinstance(xy, (tuple, list)) else xy
    vector = np.asarray(vector)
    nc = len(spaces.closed)
    no = len(spaces.open)
    nv = len(spaces.virtual)
    expected = (nc + no, no + nv)
    if vector.size != expected[0] * expected[1]:
        raise ValueError(
            "deltaS=-1 amplitude has size %d; expected %d" %
            (vector.size, expected[0] * expected[1])
        )
    vector = vector.reshape(expected)
    return spaces, SpinLoweringAmplitudes(
        co=vector[:nc, :no],
        cv=vector[:nc, no:],
        oo=vector[nc:, :no],
        ov=vector[nc:, no:],
    )


def spin_lowering_transition_densities(tdobj, xy):
    """Directed alpha-occupied to beta-target transition densities."""
    spaces, amp = split_spin_lowering(tdobj, xy)
    return spaces, amp, {
        "CO": pair_density(spaces.c_open, amp.co.T, spaces.c_closed),
        "CV": pair_density(spaces.c_virtual, amp.cv.T, spaces.c_closed),
        "OO": pair_density(spaces.c_open, amp.oo.T, spaces.c_open),
        "OV": pair_density(spaces.c_virtual, amp.ov.T, spaces.c_open),
    }


def spin_lowering_block_data(spaces, amplitudes):
    """MO index/factor map for variations of lowering transition densities."""
    return {
        "CO": (spaces.open, spaces.closed, amplitudes.co.T),
        "CV": (spaces.virtual, spaces.closed, amplitudes.cv.T),
        "OO": (spaces.open, spaces.open, amplitudes.oo.T),
        "OV": (spaces.virtual, spaces.open, amplitudes.ov.T),
    }


# Complete lowering scalar and M-matrix ledger

def spin_lowering_response_terms(spin):
    """Directed ``vref0/vref1`` coefficients in ``gen_rohf_response_sfd``."""
    denominator = 2.0 * spin - 1.0
    a = np.sqrt((2.0 * spin + 1.0) / (2.0 * spin))
    b = np.sqrt(2.0 * spin / denominator)
    c = np.sqrt((2.0 * spin + 1.0) / denominator)
    return (
        ResponseTerm("CO", "CO", 1.0, 1.0 / denominator),
        ResponseTerm("CO", "CV", a, 0.0),
        ResponseTerm("CO", "OO", b, 0.0),
        ResponseTerm(
            "CO", "OV", 2.0 * spin / denominator,
            -1.0 / denominator,
        ),
        ResponseTerm("CV", "CO", a, 0.0),
        ResponseTerm("CV", "CV", 1.0, 0.0),
        ResponseTerm("CV", "OO", c, 0.0),
        ResponseTerm("CV", "OV", a, 0.0),
        ResponseTerm("OO", "CO", b, 0.0),
        ResponseTerm("OO", "CV", c, 0.0),
        ResponseTerm("OO", "OO", 1.0, 0.0),
        ResponseTerm("OO", "OV", b, 0.0),
        ResponseTerm(
            "OV", "CO", 2.0 * spin / denominator,
            -1.0 / denominator,
        ),
        ResponseTerm("OV", "CV", a, 0.0),
        ResponseTerm("OV", "OO", b, 0.0),
        ResponseTerm("OV", "OV", 1.0, 1.0 / denominator),
    )


def spin_lowering_fock0_fockz(tdobj, max_memory=None):
    """Operators used by the current lowering-channel action."""
    mf = tdobj._scf
    if max_memory is None:
        max_memory = tdobj.max_memory
    _response, fockz = gen_rohf_response_sfd(
        mf,
        mo_coeff=mf.mo_coeff,
        mo_occ=mf.mo_occ,
        hermi=0,
        max_memory=max_memory,
    )
    if tdobj.nobeta:
        dma, dmb = mf.make_rdm1()
        dm0 = 0.5 * (dma + dmb)
        fock = mf.get_fock(dm=np.array([dm0, dm0]))
    else:
        fock = mf.get_fock()
    fock0 = 0.5 * (fock.focka + fock.fockb)
    return fock0, fockz


def spin_lowering_fock_projections(tdobj, xy):
    """Complete explicit-Fock ledger of ``X.T A_sfd X``."""
    spaces, amplitudes = split_spin_lowering(tdobj, xy)
    c = spaces.c_closed
    o = spaces.c_open
    v = spaces.c_virtual
    block_data = (
        ("C", "O", amplitudes.co),
        ("C", "V", amplitudes.cv),
        ("O", "O", amplitudes.oo),
        ("O", "V", amplitudes.ov),
    )
    orbital_data = {
        "C": (spaces.closed, c),
        "O": (spaces.open, o),
        "V": (spaces.virtual, v),
    }
    terms = []

    def add(name, left_label, coefficient, right_label, f0, fz):
        left_indices, left_orbitals = orbital_data[left_label]
        right_indices, right_orbitals = orbital_data[right_label]
        coefficient = np.asarray(coefficient)
        if coefficient.size:
            terms.append(FockProjection(
                name=name,
                left_indices=left_indices,
                left_orbitals=left_orbitals,
                coefficient=coefficient,
                right_indices=right_indices,
                right_orbitals=right_orbitals,
                weight_f0=float(f0),
                weight_fz=float(fz),
            ))

    # Ordinary alpha-to-beta spin-flip Fock difference.
    for row_left, column_left, x_left in block_data:
        for row_right, column_right, x_right in block_data:
            if row_left == row_right:
                add(
                    "base-beta-%s%s-%s%s" % (
                        row_left, column_left, row_right, column_right,
                    ),
                    column_left,
                    x_left.T @ x_right,
                    column_right,
                    1.0,
                    -1.0,
                )
            if column_left == column_right:
                add(
                    "base-alpha-%s%s-%s%s" % (
                        row_left, column_left, row_right, column_right,
                    ),
                    row_right,
                    -(x_right @ x_left.T),
                    row_left,
                    1.0,
                    1.0,
                )

    # Tensor spin-adaptation correction, expressed in the same F0/Fz basis.
    spin = spaces.spin
    trace_oo = float(np.trace(amplitudes.oo))
    eta = np.sqrt((2.0 * spin + 1.0) / (2.0 * spin)) - 1.0
    gamma = np.sqrt((2.0 * spin + 1.0) / (2.0 * spin - 1.0))
    zeta = np.sqrt(2.0 * spin / (2.0 * spin - 1.0)) - 1.0
    chi = 1.0 / np.sqrt(2.0 * spin * (2.0 * spin - 1.0))
    t_cc = (
        amplitudes.cv @ amplitudes.cv.T / spin
        + amplitudes.co @ amplitudes.co.T * 2.0 / (2.0 * spin - 1.0)
    )
    t_vv = (
        amplitudes.cv.T @ amplitudes.cv / spin
        + amplitudes.ov.T @ amplitudes.ov * 2.0 / (2.0 * spin - 1.0)
    )
    t_cv = gamma * (1.0 + 1.0 / spin) * trace_oo * amplitudes.cv
    t_beta_vo = (
        2.0 * eta * amplitudes.cv.T @ amplitudes.co
        + 2.0 * zeta * amplitudes.ov.T @ amplitudes.oo
    )
    t_beta_co = 2.0 * chi * trace_oo * amplitudes.co
    t_alpha_oc = (
        -2.0 * eta * amplitudes.cv @ amplitudes.ov.T
        - 2.0 * zeta * amplitudes.co @ amplitudes.oo.T
    ).T
    t_alpha_vo = -2.0 * chi * trace_oo * amplitudes.ov.T

    add("adapt-spin-cc", "C", t_cc, "C", 0.0, -1.0)
    add("adapt-spin-vv", "V", t_vv, "V", 0.0, -1.0)
    add("adapt-spin-cv", "C", t_cv, "V", 0.0, -1.0)
    add("adapt-beta-vo", "V", t_beta_vo, "O", 1.0, -1.0)
    add("adapt-beta-co", "C", t_beta_co, "O", 1.0, -1.0)
    add("adapt-alpha-oc", "O", t_alpha_oc, "C", 1.0, 1.0)
    add("adapt-alpha-vo", "V", t_alpha_vo, "O", 1.0, 1.0)
    return tuple(terms)


def spin_lowering_fock_probes(tdobj, xy):
    """AO probes of the channel's explicit Fock scalar."""
    return fock_probes(tdobj, spin_lowering_fock_projections(tdobj, xy))


def spin_lowering_fock_scalar(tdobj, xy, max_memory=None):
    fock0, fockz = spin_lowering_fock0_fockz(
        tdobj, max_memory=max_memory,
    )
    p0, pz = spin_lowering_fock_probes(tdobj, xy)
    return float(
        lib.einsum("pq,pq->", p0, fock0)
        + lib.einsum("pq,pq->", pz, fockz)
    )


def spin_lowering_response_scalar(tdobj, xy, max_memory=None):
    spaces, _amplitudes, densities = spin_lowering_transition_densities(
        tdobj, xy,
    )
    vref0, vref1 = _apply_reference_responses(
        tdobj, densities, max_memory=max_memory,
    )
    value = 0.0
    for term in spin_lowering_response_terms(spaces.spin):
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


def spin_lowering_ledger_scalar(tdobj, xy, max_memory=None):
    """Independent reconstruction of ``X.T gen_vind_sfd(X)``."""
    return (
        spin_lowering_fock_scalar(tdobj, xy, max_memory=max_memory)
        + spin_lowering_response_scalar(tdobj, xy, max_memory=max_memory)
    )


def spin_lowering_action_scalar(tdobj, xy):
    vector = xy[0] if isinstance(xy, (tuple, list)) else xy
    vector = np.asarray(vector)
    vind, _diagonal = tdobj.gen_vind_sfd()
    action = vind(vector.reshape(1, -1)).reshape(vector.shape)
    return float(np.vdot(vector, action).real)


# Channel assembly

def grad_elec(
        gradient_driver, tdobj, xy, atmlst=None, tolerance=1e-12,
        max_cycle=None):
    """Build the analytic excitation gradient for deltaS=-1."""
    if tdobj.deltaS != -1:
        raise ValueError("deltaS=-1 gradient received a different spin channel")
    spaces, amplitudes, densities = spin_lowering_transition_densities(
        tdobj, xy,
    )
    blocks = spin_lowering_block_data(spaces, amplitudes)
    response_terms = spin_lowering_response_terms(spaces.spin)
    channel_data = (spaces, amplitudes, densities, blocks, response_terms)
    projections = spin_lowering_fock_projections(tdobj, xy)
    probes = fock_probes(tdobj, projections)
    return assemble_gradient(
        gradient_driver, tdobj, channel_data, probes,
        fock_projection_q(tdobj, projections, spin_lowering_fock0_fockz(tdobj), probes),
        response_projection_q(tdobj, channel_data, hfx_only=True),
        atmlst=atmlst, tolerance=tolerance, max_cycle=max_cycle,
    )
