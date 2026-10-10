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

"""Shared orbital, reference-response and J/K derivative operations for NTTDA.

Spin-channel coefficients and amplitude projections stay in the channel modules.
"""

from dataclasses import dataclass

import numpy as np
from pyscf import dft, lib
from nest.nttda import nttda as nttda_mod
from nest.aocscf import AverageOccupationROKS, SymAdaptedAverageOccupationROKS


@dataclass(frozen=True)
class OrbitalSpaces:
    """Closed, open, and virtual spatial-orbital partitions."""

    closed: np.ndarray
    open: np.ndarray
    virtual: np.ndarray
    c_closed: np.ndarray
    c_open: np.ndarray
    c_virtual: np.ndarray

    @property
    def spin(self):
        return 0.5 * len(self.open)


def orbital_spaces(tdobj):
    """Return the ROKS ``C/O/V`` orbital partition used by NTTDA."""
    mf = tdobj._scf
    occ = np.asarray(mf.mo_occ)
    if occ.ndim != 1:
        raise ValueError("NTTDA gradients require spatial ROKS orbitals")
    closed, open_, virtual = nttda_mod._orbital_indices(tdobj)
    coeff = np.asarray(mf.mo_coeff)
    return OrbitalSpaces(
        closed=closed,
        open=open_,
        virtual=virtual,
        c_closed=coeff[:, closed],
        c_open=coeff[:, open_],
        c_virtual=coeff[:, virtual],
    )


def pair_density(c_left, coefficient, c_right):
    """Build ``C_left coefficient C_right^T`` without symmetrizing it."""
    return c_left @ np.asarray(coefficient) @ c_right.conj().T


@dataclass(frozen=True)
class FockProjection:
    """One scalar term ``Tr[P (weight_f0 F0 + weight_fz Fz)]``."""

    name: str
    left_indices: np.ndarray
    coefficient: np.ndarray
    right_indices: np.ndarray
    weight_f0: float
    weight_fz: float


def _fock_response_q(tdobj, p_alpha, p_beta):
    """Reference-density derivative of a spin-resolved Fock scalar."""
    mf = tdobj._scf
    mo = np.asarray(mf.mo_coeff)
    if isinstance(mf, (AverageOccupationROKS, SymAdaptedAverageOccupationROKS)):
        occupation = np.asarray(mf.mo_occ)
        probe = np.asarray(p_alpha) + np.asarray(p_beta)
        potential = lib.view(mf, dft.rks.RKS).gen_response(hermi=0)(probe.T)
        q_total = (
            mo.conj().T @ (potential + potential.T) @ mo
        ) * occupation[None, :]
        return 0.5 * q_total, 0.5 * q_total
    occ_alpha = (np.asarray(mf.mo_occ) > 0).astype(float)
    occ_beta = (np.asarray(mf.mo_occ) == 2).astype(float)
    if (isinstance(mf, dft.KohnShamDFT)
            and mf._numint._xc_type(mf.xc) != "HF"):
        unrestricted = mf.to_uks()
        unrestricted.verbose = 0
        v_alpha, v_beta = unrestricted.gen_response(hermi=0)(
            np.asarray((p_alpha.T, p_beta.T))
        )
    else:
        p_total = p_alpha + p_beta
        coulomb = mf.get_j(mf.mol, p_total.T, hermi=0)
        v_alpha = coulomb - mf.get_k(mf.mol, p_alpha.T, hermi=0)
        v_beta = coulomb - mf.get_k(mf.mol, p_beta.T, hermi=0)
    q_alpha = (
        mo.conj().T @ (v_alpha + v_alpha.T) @ mo
    ) * occ_alpha[None, :]
    q_beta = (
        mo.conj().T @ (v_beta + v_beta.T) @ mo
    ) * occ_beta[None, :]
    return q_alpha, q_beta


@dataclass(frozen=True)
class ResponseTerm:
    """Directed response term from one source density to one target block."""

    target: str
    source: str
    vref0: float
    vref1: float


def _fxc_reference(tdobj):
    mf = tdobj._scf
    ni = mf._numint
    fxc = ni.cache_xc_kernel(
        mf.mol, mf.grids, mf.xc, mf.mo_coeff, mf.mo_occ, 1,
    )[2]
    return 0.5 * (
        fxc[0, :, 0] - fxc[0, :, 1]
        - fxc[1, :, 0] + fxc[1, :, 1]
    )


def _apply_reference_responses(tdobj, densities, max_memory=None):
    """Return separate ``vref0`` and ``vref1`` actions for each density."""
    mf = tdobj._scf
    mol = mf.mol
    ni = mf._numint
    if max_memory is None:
        max_memory = tdobj.max_memory
    labels = tuple(densities)
    dms = np.asarray([densities[label] for label in labels])
    xctype = ni._xc_type(mf.xc)
    if xctype == "HF":
        vref0 = np.zeros_like(dms)
        vref1 = np.zeros_like(dms)
    else:
        fxc_ref = _fxc_reference(tdobj)
        vref0 = ni.nr_rks_fxc(
            mol, mf.grids, mf.xc, None, dms, 0, 0,
            None, None, fxc_ref, max_memory=max_memory,
        )
        if xctype == "LDA":
            vref1 = vref0.copy()
        elif xctype == "GGA":
            vref1 = nttda_mod.nr_rks_fxc1_gga(
                ni, mol, mf.grids, mf.xc, dms, fxc_ref,
                max_memory=max_memory,
            )
        elif xctype == "MGGA":
            vref1 = nttda_mod.nr_rks_fxc1_mgga(
                ni, mol, mf.grids, mf.xc, dms, fxc_ref,
                max_memory=max_memory,
            )
        else:
            raise NotImplementedError(
                "NTTDA response does not support XC type %s" % xctype
            )

    omega, alpha, hyb = ni.rsh_and_hybrid_coeff(mf.xc, mol.spin)
    if ni.libxc.is_hybrid_xc(mf.xc):
        vref0 -= hyb * mf.get_k(mol, dms, hermi=0)
        vref1 -= hyb * mf.get_j(mol, dms, hermi=0)
        if omega != 0:
            scale = alpha - hyb
            vref0 -= scale * mf.get_k(mol, dms, hermi=0, omega=omega)
            vref1 -= scale * mf.get_j(mol, dms, hermi=0, omega=omega)
    return (
        {label: value for label, value in zip(labels, vref0)},
        {label: value for label, value in zip(labels, vref1)},
    )


def _apply_hfx_responses(tdobj, densities):
    """Return only the hybrid/RSH J/K portions of ``vref0/vref1``."""
    mf = tdobj._scf
    labels = tuple(densities)
    dms = np.asarray([densities[label] for label in labels])
    vref0 = np.zeros_like(dms)
    vref1 = np.zeros_like(dms)
    ni = mf._numint
    omega, alpha, hybrid = ni.rsh_and_hybrid_coeff(mf.xc, mf.mol.spin)
    if ni.libxc.is_hybrid_xc(mf.xc):
        vref0 -= hybrid * mf.get_k(mf.mol, dms, hermi=0)
        vref1 -= hybrid * mf.get_j(mf.mol, dms, hermi=0)
        if omega != 0:
            scale = alpha - hybrid
            vref0 -= scale * mf.get_k(
                mf.mol, dms, hermi=0, omega=omega,
            )
            vref1 -= scale * mf.get_j(
                mf.mol, dms, hermi=0, omega=omega,
            )
    return (
        {label: value for label, value in zip(labels, vref0)},
        {label: value for label, value in zip(labels, vref1)},
    )


def _as_derivative_stack(array):
    array = np.asarray(array)
    if array.ndim == 3:
        array = array[None]
    return array


def _density_key(density):
    density = np.asarray(density)
    data = density.__array_interface__["data"][0]
    return data, density.shape, density.strides, density.dtype.str


@dataclass(frozen=True)
class JKDerivativeTerm:
    """One fixed-AO bilinear derivative: probe 0 is direct, probe 1 is orbital."""

    left: np.ndarray
    right: np.ndarray
    scale: float
    omega: float | None
    probe: int = 0


def contract_jk_derivatives(gradient_driver, mol, atoms, j_terms, k_terms):
    """Contract both derivative lists into [direct, Z-vector] gradients.

    Range separation, density reuse and memory-limited batches are handled
    here. Callers receive completed contractions; no terms are retained.
    """
    gradients = np.zeros((2, len(atoms), 3))
    offsets = mol.offset_nr_by_atom()
    for operator, terms in (("j", j_terms), ("k", k_terms)):
        nonzero_terms = [term for term in terms if term.scale != 0.0]
        _contract_derivative_terms(
            gradients, gradient_driver, mol, atoms, offsets, nonzero_terms, operator,
        )
    return gradients


def _term_densities(term, exchange):
    left, right = term.left, term.right
    if exchange:
        return left, right, left.T, right.T
    return left, right


def _density_batches(terms, exchange, max_memory, nao):
    """Group bilinear terms while bounding derivative-potential storage."""
    minimum = 4 if exchange else 2
    bytes_per_density = 4 * nao * nao * np.dtype(float).itemsize
    batch_limit = max(
        minimum,
        int(0.2 * max_memory * 1e6 / bytes_per_density),
    )
    batch = []
    keys = set()
    for term in terms:
        term_keys = {
            _density_key(density)
            for density in _term_densities(term, exchange)
        }
        if batch and len(keys | term_keys) > batch_limit:
            yield batch
            batch = []
            keys = set()
        batch.append(term)
        keys.update(term_keys)
    if batch:
        yield batch


def _jk_derivative_potentials(
        gradient_driver, mol, terms, operator, omega):
    exchange = operator == "k"
    densities = {}
    for term in terms:
        for density in _term_densities(term, exchange):
            density = np.asarray(density)
            densities.setdefault(_density_key(density), density)
    keys = tuple(densities)
    stack = np.asarray([densities[key] for key in keys])
    get_potential = gradient_driver.get_j if operator == "j" else gradient_driver.get_k
    values = get_potential(mol, stack, hermi=0, omega=omega)
    values = _as_derivative_stack(values)
    return dict(zip(keys, values))


def _contract_derivative_terms(
        gradients, gradient_driver, mol, atoms, offsets, terms,
        operator):
    if not atoms:
        return
    terms_by_omega = {}
    for term in terms:
        terms_by_omega.setdefault(term.omega, []).append(term)
    exchange = operator == "k"
    for omega, omega_terms in terms_by_omega.items():
        for batch in _density_batches(
                omega_terms, exchange, gradient_driver.max_memory,
                mol.nao_nr()):
            potentials = _jk_derivative_potentials(
                gradient_driver, mol, batch, operator, omega,
            )
            for term in batch:
                left = np.asarray(term.left)
                right = np.asarray(term.right)
                right_derivative = potentials[_density_key(right)]
                left_derivative = potentials[_density_key(left)]
                if exchange:
                    right_t_derivative = potentials[
                        _density_key(right.T)
                    ]
                    left_t_derivative = potentials[_density_key(left.T)]
                for k, atom in enumerate(atoms):
                    p0, p1 = offsets[atom][2:]
                    if exchange:
                        value = lib.einsum(
                            "xpq,pq->x",
                            right_derivative[:, p0:p1, :],
                            left[p0:p1, :],
                        )
                        value += lib.einsum(
                            "xqp,pq->x",
                            right_t_derivative[:, p0:p1, :],
                            left[:, p0:p1],
                        )
                        value += lib.einsum(
                            "xpq,pq->x",
                            left_derivative[:, p0:p1, :],
                            right[p0:p1, :],
                        )
                        value += lib.einsum(
                            "xqp,pq->x",
                            left_t_derivative[:, p0:p1, :],
                            right[:, p0:p1],
                        )
                    else:
                        value = lib.einsum(
                            "xpq,pq->x",
                            right_derivative[:, p0:p1],
                            left[p0:p1],
                        )
                        value += lib.einsum(
                            "xpq,qp->x",
                            right_derivative[:, p0:p1],
                            left[:, p0:p1],
                        )
                        value += lib.einsum(
                            "xpq,pq->x",
                            left_derivative[:, p0:p1],
                            right[p0:p1],
                        )
                        value += lib.einsum(
                            "xpq,qp->x",
                            left_derivative[:, p0:p1],
                            right[:, p0:p1],
                        )
                    gradients[term.probe][k] += term.scale * value


def response_jk_terms(tdobj, densities, response_terms):
    """Build explicit J/K derivative terms for the channel response scalar."""
    mf = tdobj._scf
    ni = mf._numint
    if not ni.libxc.is_hybrid_xc(mf.xc):
        return [], []
    omega, alpha, hybrid = ni.rsh_and_hybrid_coeff(mf.xc, mf.mol.spin)
    scales = [(hybrid, None)]
    if omega != 0:
        scales.append((alpha - hybrid, omega))
    j_terms, k_terms = [], []
    for term in response_terms:
        target, source = densities[term.target], densities[term.source]
        for coefficient, range_omega in scales:
            if term.vref0:
                k_terms.append(JKDerivativeTerm(
                    target, source, -coefficient * term.vref0, range_omega,
                ))
            if term.vref1:
                j_terms.append(JKDerivativeTerm(
                    target, source, -coefficient * term.vref1, range_omega,
                ))
    return j_terms, k_terms


def fockz_hfx_terms(tdobj, pz):
    """Return MO derivatives and K terms of ``-1/2 Pz:K(D_OO)``."""
    mf = tdobj._scf
    mol = mf.mol
    ni = mf._numint
    mo = np.asarray(mf.mo_coeff)
    q_alpha = np.zeros((mo.shape[1], mo.shape[1]))
    q_beta = np.zeros_like(q_alpha)
    if not ni.libxc.is_hybrid_xc(mf.xc):
        return q_alpha, q_beta, []

    spaces = orbital_spaces(tdobj)
    density_open = spaces.c_open @ spaces.c_open.T
    omega, alpha, hybrid = ni.rsh_and_hybrid_coeff(mf.xc, mol.spin)
    scales = [(hybrid, None)]
    if omega != 0:
        scales.append((alpha - hybrid, omega))
    k_terms = []
    for coefficient, range_omega in scales:
        if coefficient == 0.0:
            continue
        if range_omega is None:
            potential = mf.get_k(mol, pz, hermi=0)
        else:
            potential = mf.get_k(
                mol, pz, hermi=0, omega=range_omega,
            )
        q_alpha[:, spaces.open] -= 0.5 * coefficient * (
            mo.conj().T @ (potential + potential.T) @ spaces.c_open
        )
        k_terms.append(JKDerivativeTerm(
            pz, density_open, -0.5 * coefficient, range_omega,
        ))
    return q_alpha, q_beta, k_terms


def fock_probes(tdobj, projections):
    """AO probes of the explicit F0/Fz scalar for any spin channel."""
    mo = np.asarray(tdobj._scf.mo_coeff)
    p0 = np.zeros((tdobj.mol.nao_nr(), tdobj.mol.nao_nr()))
    pz = np.zeros_like(p0)
    for term in projections:
        density = pair_density(
            mo[:, term.left_indices], term.coefficient, mo[:, term.right_indices],
        )
        p0 += term.weight_f0 * density
        pz += term.weight_fz * density
    return p0, pz


def fock_projection_q(tdobj, projections, operators, probes):
    """Differentiate the Fock projections and the reference density.

    HF spin probes include the full Fz response. DFT adds Fz and nobeta
    corrections during gradient assembly, after the common F0 response.
    """
    mf = tdobj._scf
    mo = np.asarray(mf.mo_coeff)
    fock0, fockz = (mo.conj().T @ operator @ mo for operator in operators)
    q_alpha = np.zeros((mo.shape[1], mo.shape[1]))
    q_beta = np.zeros_like(q_alpha)
    is_hf = mf._numint._xc_type(mf.xc) == "HF"
    for term in projections:
        left, right = term.left_indices, term.right_indices
        coefficient = term.coefficient
        def project(target, operator, scale):
            if scale:
                target[:, left] += scale * operator[:, right] @ coefficient.T
                target[:, right] += scale * operator[:, left] @ coefficient
        project(q_alpha, fock0, 0.5 * term.weight_f0)
        project(q_beta, fock0, 0.5 * term.weight_f0)
        if is_hf:
            project(q_alpha, fockz, 0.5 * term.weight_fz)
            project(q_beta, fockz, 0.5 * term.weight_fz)
        else:
            project(q_alpha, fockz, term.weight_fz)
    p0, pz = probes
    p_alpha, p_beta = 0.5 * p0, 0.5 * p0
    if is_hf:
        p_alpha = p_alpha + 0.5 * pz
        p_beta = p_beta - 0.5 * pz
    response_alpha, response_beta = _fock_response_q(tdobj, p_alpha, p_beta)
    return q_alpha + response_alpha, q_beta + response_beta


def response_potentials(densities, vref0, vref1, terms):
    """Vary both transition-density factors of the response scalar."""
    potentials = {label: np.zeros_like(dm) for label, dm in densities.items()}
    for term in terms:
        if term.vref0:
            potentials[term.target] += term.vref0 * vref0[term.source]
            potentials[term.source] += term.vref0 * vref0[term.target]
        if term.vref1:
            potentials[term.target] += term.vref1 * vref1[term.source]
            potentials[term.source] += term.vref1 * vref1[term.target]
    return potentials


def hfx_response_q(tdobj, densities, blocks, terms):
    """MO derivative of the hybrid/RSH response at fixed kernels."""
    vref0, vref1 = _apply_hfx_responses(tdobj, densities)
    potentials = response_potentials(densities, vref0, vref1, terms)
    return project_channel_potentials(tdobj, potentials, blocks)


def project_channel_potentials(tdobj, potentials, blocks):
    """Project transition-factor potentials for any NTTDA spin channel."""
    mo = np.asarray(tdobj._scf.mo_coeff)
    q_alpha = np.zeros((mo.shape[1], mo.shape[1]))
    q_beta = np.zeros_like(q_alpha)
    for label, (target, source, coefficient) in blocks.items():
        potential = mo.conj().T @ potentials[label] @ mo
        q_beta[:, target] += potential[:, source] @ coefficient.T
        q_alpha[:, source] += potential[target, :].T @ coefficient
    return q_alpha, q_beta

def reference_spin_densities(tdobj):
    """Spin densities of the variational reference used by the XC kernel."""
    mf = tdobj._scf
    mo = np.asarray(mf.mo_coeff)
    if isinstance(mf, (AverageOccupationROKS, SymAdaptedAverageOccupationROKS)):
        density = (mo * mf.mo_occ) @ mo.conj().T
        return 0.5 * density, 0.5 * density
    return (
        mo[:, mf.mo_occ > 0] @ mo[:, mf.mo_occ > 0].T,
        mo[:, mf.mo_occ == 2] @ mo[:, mf.mo_occ == 2].T,
    )

def reference_spin_occupations(tdobj):
    """Per-orbital alpha/beta occupations of the reference density."""
    mf = tdobj._scf
    occupation = np.asarray(mf.mo_occ)
    if isinstance(mf, (AverageOccupationROKS, SymAdaptedAverageOccupationROKS)):
        return 0.5 * occupation, 0.5 * occupation
    return (occupation > 0).astype(float), (occupation == 2).astype(float)
