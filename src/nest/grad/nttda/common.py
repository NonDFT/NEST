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
from pyscf.grad import rhf as rhf_grad
from nest.nttda import nttda as nttda_mod

from . import xc as xc_backend
from .xc import _reference_spin_densities
from .roks import finish_gradient


def assemble_gradient(
        gradient_driver, tdobj, channel_data, probes, fock_q, response_q,
        atmlst=None, tolerance=1e-12, max_cycle=None):
    """Assemble channel projections, XC/J/K derivatives and the adjoint.

    ``response_q`` contains only the hybrid/RSH part; semilocal response is
    added here. The direct and Z-vector probes share one J/K derivative batch.
    """
    mf = tdobj._scf
    xctype = mf._numint._xc_type(mf.xc)
    atmlst = tuple(range(tdobj.mol.natm) if atmlst is None else atmlst)
    spaces, _amplitudes, densities, _blocks, response_terms = channel_data
    p0, pz = probes
    fock_alpha, fock_beta = fock_q
    response_alpha, response_beta = response_q
    m_matrix = fock_alpha + fock_beta + response_alpha + response_beta
    ledger = _JKDerivativeLedger()
    slots = ("direct", "zvector")
    direct = response_direct_hfx(
        gradient_driver, tdobj, densities, response_terms,
        atmlst=atmlst, jk_ledger=ledger, output_slot=slots[0],
    )

    # ROKS/HF includes Fz in the spin-resolved Fock probes. All other
    # references use charge-only probes and differentiate Fz separately.
    spin_fock = xctype == "HF" and not nttda_mod._is_average_occupation_reference(mf)
    direct_fock_probes = (
        (0.5 * (p0 + pz), 0.5 * (p0 - pz)) if spin_fock
        else (0.5 * p0, 0.5 * p0)
    )
    if xctype != "HF":
        for terms in (
                xc_backend.response_terms(gradient_driver, tdobj, channel_data, atmlst=atmlst),
                xc_backend.fockz_terms(gradient_driver, tdobj, spaces, pz, atmlst=atmlst)):
            m_matrix += terms.q_alpha + terms.q_beta
            direct += terms.direct
        common_alpha, common_beta = xc_backend.nobeta_reference_q(tdobj, p0)
        m_matrix += common_alpha + common_beta
    if not spin_fock:
        terms = fockz_hfx_terms(
            gradient_driver, tdobj, pz, atmlst=atmlst,
            jk_ledger=ledger, output_slot=slots[0],
        )
        m_matrix += terms.q_alpha + terms.q_beta
        direct += terms.direct

    def fock_direct(driver, obj, p_alpha, p_beta, atmlst=None):
        local = spin_fock_direct(
            driver, obj, p_alpha, p_beta, atmlst=atmlst, nobeta_p0=p0,
            jk_ledger=ledger, output_slots=slots,
        )
        contractions = ledger.contract(driver, obj.mol, atmlst, slots=slots)
        for index, slot in enumerate(slots):
            local[index] += contractions[slot]
        return local

    return finish_gradient(
        gradient_driver, tdobj, m_matrix, direct, atmlst, tolerance,
        max_cycle, fock_direct, direct_fock_probes=direct_fock_probes,
    )

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
    closed = np.flatnonzero(occ == 2)
    open_ = np.flatnonzero(occ == 1)
    virtual = np.flatnonzero(occ == 0)
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
    left_orbitals: np.ndarray
    coefficient: np.ndarray
    right_indices: np.ndarray
    right_orbitals: np.ndarray
    weight_f0: float
    weight_fz: float

    def density(self):
        return pair_density(
            self.left_orbitals, self.coefficient, self.right_orbitals,
        )


def _fock_response_q(tdobj, p_alpha, p_beta):
    """Reference-density derivative of a spin-resolved Fock scalar."""
    mf = tdobj._scf
    mo = np.asarray(mf.mo_coeff)
    if nttda_mod._is_average_occupation_reference(mf):
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
class _JKDerivativeTerm:
    """One fixed-AO bilinear derivative with a named output slot."""

    left: np.ndarray
    right: np.ndarray
    scale: float
    omega: float
    slot: object


class _JKDerivativeLedger:
    """Shared scheduler for fixed-AO J/K derivative contractions."""

    def __init__(self):
        self._terms = {"j": [], "k": []}

    def add(self, operator, slot, terms):
        self._terms[operator].extend(
            _JKDerivativeTerm(left, right, scale, omega, slot)
            for left, right, scale, omega in terms
            if scale != 0.0
        )

    def contract(self, gradient_driver, mol, atoms, slots=()):
        atoms = tuple(atoms)
        shape = (len(atoms), 3)
        gradients = {slot: np.zeros(shape) for slot in slots}
        for operator in ("j", "k"):
            for term in self._terms[operator]:
                gradients.setdefault(term.slot, np.zeros(shape))
            _contract_derivative_terms(
                gradients,
                gradient_driver,
                mol,
                atoms,
                mol.offset_nr_by_atom(),
                self._terms[operator],
                operator,
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
    if operator == "j":
        if omega is None:
            values = gradient_driver.get_j(mol, stack, hermi=0)
        else:
            values = gradient_driver.get_j(
                mol, stack, hermi=0, omega=omega,
            )
    else:
        if omega is None:
            values = gradient_driver.get_k(mol, stack, hermi=0)
        else:
            values = gradient_driver.get_k(
                mol, stack, hermi=0, omega=omega,
            )
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
                    gradients[term.slot][k] += term.scale * value


def _spin_probe_stacks(p_alpha, p_beta):
    p_alpha = np.asarray(p_alpha)
    p_beta = np.asarray(p_beta)
    single_probe = p_alpha.ndim == 2
    if single_probe:
        p_alpha = p_alpha[None]
        p_beta = p_beta[None]
    return p_alpha, p_beta, single_probe


def spin_fock_direct(
        gradient_driver, tdobj, p_alpha, p_beta, atmlst=None,
        nobeta_p0=None, jk_ledger=None, output_slots=None):
    """Differentiate one or more HF/DFT spin Fock scalar probes.

    The optional ``nobeta_p0`` correction belongs to the first, explicit-direct
    probe in the batch.
    """
    mf = tdobj._scf
    mol = tdobj.mol
    if atmlst is None:
        atmlst = range(mol.natm)
    atmlst = tuple(atmlst)
    p_alpha, p_beta, single_probe = _spin_probe_stacks(
        p_alpha, p_beta,
    )
    if output_slots is None:
        output_slots = tuple(range(len(p_alpha)))
    p_total = p_alpha + p_beta
    density_alpha, density_beta = _reference_spin_densities(tdobj)
    gradient = np.zeros((len(p_alpha), len(atmlst), 3))
    hcore_derivative = rhf_grad.Gradients(mf).hcore_generator(mol)
    for k, atom in enumerate(atmlst):
        gradient[:, k] += lib.einsum(
            "npq,xpq->nx", p_total, hcore_derivative(atom),
        )
    ni = mf._numint
    omega, alpha, hybrid = ni.rsh_and_hybrid_coeff(mf.xc, mol.spin)
    local_ledger = _JKDerivativeLedger()
    ledger = jk_ledger if jk_ledger is not None else local_ledger
    for probe in range(len(p_alpha)):
        j_terms = [
            (p_total[probe], density_alpha, 1.0, None),
            (p_total[probe], density_beta, 1.0, None),
        ]
        k_terms = []
        if ni.libxc.is_hybrid_xc(mf.xc):
            k_terms.extend((
                (p_alpha[probe], density_alpha, -hybrid, None),
                (p_beta[probe], density_beta, -hybrid, None),
            ))
            if omega != 0:
                long_range = -(alpha - hybrid)
                k_terms.extend((
                    (p_alpha[probe], density_alpha, long_range, omega),
                    (p_beta[probe], density_beta, long_range, omega),
                ))
        ledger.add("j", output_slots[probe], j_terms)
        ledger.add("k", output_slots[probe], k_terms)
    xctype = ni._xc_type(mf.xc)
    if xctype != "HF":
        if xctype == "LDA":
            derivative_contractor = xc_backend.contract_lda_vxc_derivative
        elif xctype == "GGA":
            derivative_contractor = xc_backend.contract_gga_vxc_derivative
        elif xctype == "MGGA":
            derivative_contractor = xc_backend.contract_mgga_vxc_derivative
        else:
            raise NotImplementedError(
                "ordinary Fock direct derivative is not implemented for %s" %
                xctype
            )
        if (nobeta_p0 is not None and tdobj.nobeta
                and not nttda_mod._is_average_occupation_reference(mf)):
            density0 = 0.5 * (density_alpha + density_beta)
            actual_probe_alpha = np.array(p_alpha, copy=True)
            actual_probe_beta = np.array(p_beta, copy=True)
            actual_probe_alpha[0] -= 0.5 * nobeta_p0
            actual_probe_beta[0] -= 0.5 * nobeta_p0
        else:
            density0 = None
            actual_probe_alpha = p_alpha
            actual_probe_beta = p_beta
        gradient += derivative_contractor(
            mf,
            density_alpha,
            density_beta,
            actual_probe_alpha,
            actual_probe_beta,
            atmlst=atmlst,
            max_memory=gradient_driver.max_memory,
        )
        if density0 is not None:
            gradient[0] += derivative_contractor(
                mf,
                density0,
                density0,
                0.5 * nobeta_p0,
                0.5 * nobeta_p0,
                atmlst=atmlst,
                max_memory=gradient_driver.max_memory,
            )
    if jk_ledger is None:
        contractions = local_ledger.contract(
            gradient_driver, mol, atmlst, slots=output_slots,
        )
        for probe, slot in enumerate(output_slots):
            gradient[probe] += contractions[slot]
    return gradient[0] if single_probe else gradient


def response_direct_hfx(
        gradient_driver, tdobj, densities, response_terms, atmlst=None,
        jk_ledger=None, output_slot=0):
    """J/K skeleton derivative for a channel response-term ledger."""
    mol = tdobj.mol
    if atmlst is None:
        atmlst = range(mol.natm)
    atmlst = tuple(atmlst)
    gradient = np.zeros((len(atmlst), 3))
    ni = tdobj._scf._numint
    omega, alpha, hybrid = ni.rsh_and_hybrid_coeff(
        tdobj._scf.xc, mol.spin,
    )
    if not ni.libxc.is_hybrid_xc(tdobj._scf.xc):
        return gradient

    scales = [(hybrid, None)]
    if omega != 0:
        scales.append((alpha - hybrid, omega))
    j_terms = []
    k_terms = []
    for term in response_terms:
        target = densities[term.target]
        source = densities[term.source]
        for coefficient, range_omega in scales:
            if term.vref0:
                k_terms.append((
                    target,
                    source,
                    -coefficient * term.vref0,
                    range_omega,
                ))
            if term.vref1:
                j_terms.append((
                    target,
                    source,
                    -coefficient * term.vref1,
                    range_omega,
                ))
    local_ledger = _JKDerivativeLedger()
    ledger = jk_ledger if jk_ledger is not None else local_ledger
    ledger.add("j", output_slot, j_terms)
    ledger.add("k", output_slot, k_terms)
    if jk_ledger is None:
        gradient += local_ledger.contract(
            gradient_driver, mol, atmlst, slots=(output_slot,),
        )[output_slot]
    return gradient


def fockz_hfx_terms(
        gradient_driver, tdobj, pz, atmlst=None, with_direct=True,
        jk_ledger=None, output_slot=0):
    """Differentiate ``-1/2 Pz:K(D_OO)`` excluding the Pz projection."""
    mf = tdobj._scf
    mol = mf.mol
    ni = mf._numint
    if atmlst is None:
        atmlst = range(mol.natm)
    atmlst = tuple(atmlst)
    mo = np.asarray(mf.mo_coeff)
    q_alpha = np.zeros((mo.shape[1], mo.shape[1]))
    q_beta = np.zeros_like(q_alpha)
    direct = np.zeros((len(atmlst), 3))
    if not ni.libxc.is_hybrid_xc(mf.xc):
        return xc_backend.XCGradientTerms(q_alpha, q_beta, direct)

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
        if with_direct:
            k_terms.append((
                pz,
                density_open,
                -0.5 * coefficient,
                range_omega,
            ))
    if with_direct:
        local_ledger = _JKDerivativeLedger()
        ledger = jk_ledger if jk_ledger is not None else local_ledger
        ledger.add("k", output_slot, k_terms)
        if jk_ledger is None:
            direct += local_ledger.contract(
                gradient_driver, mol, atmlst, slots=(output_slot,),
            )[output_slot]
    return xc_backend.XCGradientTerms(q_alpha, q_beta, direct)


def fock_probes(tdobj, projections):
    """AO probes of the explicit F0/Fz scalar for either spin channel."""
    p0 = np.zeros((tdobj.mol.nao_nr(), tdobj.mol.nao_nr()))
    pz = np.zeros_like(p0)
    for term in projections:
        density = term.density()
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


def response_projection_q(tdobj, channel_data, max_memory=None, hfx_only=False):
    """MO derivative of the response at fixed kernels, for either channel."""
    _spaces, _amplitudes, densities, blocks, terms = channel_data
    if hfx_only:
        vref0, vref1 = _apply_hfx_responses(tdobj, densities)
    else:
        vref0, vref1 = _apply_reference_responses(tdobj, densities, max_memory)
    potentials = response_potentials(densities, vref0, vref1, terms)
    return xc_backend._project_channel_potentials(tdobj, potentials, blocks)
