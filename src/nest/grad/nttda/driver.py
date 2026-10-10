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

"""Sequential assembly of NTTDA excitation-energy gradients.

Channel modules provide amplitudes and coefficients; this module combines
MO derivatives, the orbital adjoint, and fixed-AO nuclear derivatives.
"""

from dataclasses import dataclass

import numpy as np
from pyscf import lib
from pyscf.grad import rhf as rhf_grad
from nest.aocscf import AverageOccupationROKS, SymAdaptedAverageOccupationROKS

from . import common, ensemble, roks, xc


@dataclass(frozen=True)
class GradientComponents:
    """Excitation-gradient pieces and the solved orbital adjoint."""

    m_matrix: np.ndarray
    direct: np.ndarray
    orbital: np.ndarray
    total: np.ndarray
    zvector: np.ndarray
    residual: float


def assemble_gradient(
    gradient_driver,
    tdobj,
    *,
    spaces,
    densities,
    blocks,
    response_terms,
    projections,
    operators,
    atmlst=None,
    tolerance=1e-12,
    max_cycle=None,
):
    """Build d(omega)/dR from channel data in four explicit stages.

    ``blocks`` maps each transition density to its MO indices and amplitude.
    ``projections`` and ``response_terms`` contain the channel coefficients.
    J/K terms are collected as data and contracted once, after solving Z.
    """
    mf = tdobj._scf
    xctype = mf._numint._xc_type(mf.xc)
    atmlst = tuple(range(tdobj.mol.natm) if atmlst is None else atmlst)
    p0, pz = common.fock_probes(tdobj, projections)

    # 1. Differentiate the excitation scalar with respect to MO coefficients.
    fock_alpha, fock_beta = common.fock_projection_q(
        tdobj,
        projections,
        operators,
        (p0, pz),
    )
    response_alpha, response_beta = common.hfx_response_q(
        tdobj,
        densities,
        blocks,
        response_terms,
    )
    m_matrix = fock_alpha + fock_beta + response_alpha + response_beta
    direct = np.zeros((len(atmlst), 3))
    j_terms, k_terms = common.response_jk_terms(tdobj, densities, response_terms)

    # Only ordinary ROKS/HF includes Fz in the spin-resolved Fock probes.
    ensemble_reference = isinstance(
        mf,
        (AverageOccupationROKS, SymAdaptedAverageOccupationROKS),
    )
    spin_fock = xctype == 'HF' and not ensemble_reference
    if spin_fock:
        direct_alpha, direct_beta = 0.5 * (p0 + pz), 0.5 * (p0 - pz)
    else:
        direct_alpha, direct_beta = 0.5 * p0, 0.5 * p0
    if xctype != 'HF':
        for q_alpha, q_beta, derivative in (
            xc.response_terms(gradient_driver, tdobj, densities, blocks, response_terms, atmlst=atmlst),
            xc.fockz_terms(gradient_driver, tdobj, spaces, pz, atmlst=atmlst),
        ):
            m_matrix += q_alpha + q_beta
            direct += derivative
        q_alpha, q_beta = xc.nobeta_reference_q(
            tdobj,
            p0,
            max_memory=gradient_driver.max_memory,
        )
        m_matrix += q_alpha + q_beta
    if not spin_fock:
        q_alpha, q_beta, fockz_k_terms = common.fockz_hfx_terms(tdobj, pz)
        m_matrix += q_alpha + q_beta
        k_terms.extend(fockz_k_terms)

    # 2. Solve the reference-specific transpose Hessian equation.
    response = ensemble if ensemble_reference else roks
    transpose_action, pairs = response.make_hessian_transpose_action(tdobj)
    rhs = roks.pack_m_matrix(m_matrix, pairs)
    zvector = response.solve_zvector(
        transpose_action,
        pairs,
        tdobj,
        rhs,
        tolerance=tolerance,
        max_cycle=max_cycle,
    )
    adjoint = response.zvector_adjoint_matrix(tdobj, pairs, zvector)
    residual = float(np.max(np.abs(roks.pack_m_matrix(adjoint, pairs) - rhs), initial=0.0))
    probe_alpha, probe_beta = response.zvector_probe_densities(tdobj, pairs, zvector)

    # 3. Batch the explicit and Z-vector Fock derivatives, in that order.
    fock_direct, fock_j_terms, fock_k_terms = _spin_fock_terms(
        gradient_driver,
        tdobj,
        np.asarray((direct_alpha, probe_alpha)),
        np.asarray((direct_beta, probe_beta)),
        atmlst,
        nobeta_p0=p0,
    )
    jk_direct, jk_orbital = common.contract_jk_derivatives(
        gradient_driver,
        tdobj.mol,
        atmlst,
        j_terms + fock_j_terms,
        k_terms + fock_k_terms,
    )

    # 4. Add the overlap response and retain the pieces for diagnostics.
    direct += fock_direct[0] + jk_direct
    orbital = _orbital_gradient(
        tdobj,
        m_matrix,
        adjoint,
        fock_direct[1] + jk_orbital,
        atmlst=atmlst,
    )
    return GradientComponents(
        m_matrix=m_matrix,
        direct=direct,
        orbital=orbital,
        total=direct + orbital,
        zvector=zvector,
        residual=residual,
    )


def _spin_fock_terms(gradient_driver, tdobj, p_alpha, p_beta, atmlst, nobeta_p0):
    """Return hcore/XC derivatives and the J/K terms for two Fock probes.

    Probe 0 is the explicit excitation derivative; probe 1 is the Z-vector.
    The nobeta correction belongs only to probe 0. J/K contraction is left
    to the caller so these terms can share a batch with the response terms.
    """
    mf = tdobj._scf
    mol = tdobj.mol
    p_total = p_alpha + p_beta
    density_alpha, density_beta = common.reference_spin_densities(tdobj)
    gradient = np.zeros((len(p_alpha), len(atmlst), 3))
    hcore_derivative = rhf_grad.Gradients(mf).hcore_generator(mol)
    for k, atom in enumerate(atmlst):
        gradient[:, k] += lib.einsum(
            'npq,xpq->nx',
            p_total,
            hcore_derivative(atom),
        )
    ni = mf._numint
    omega, alpha, hybrid = ni.rsh_and_hybrid_coeff(mf.xc, mol.spin)
    j_terms, k_terms = [], []
    for probe in range(len(p_alpha)):
        j_terms.extend(
            (
                common.JKDerivativeTerm(p_total[probe], density_alpha, 1.0, None, probe),
                common.JKDerivativeTerm(p_total[probe], density_beta, 1.0, None, probe),
            )
        )
        if ni.libxc.is_hybrid_xc(mf.xc):
            k_terms.extend(
                (
                    common.JKDerivativeTerm(p_alpha[probe], density_alpha, -hybrid, None, probe),
                    common.JKDerivativeTerm(p_beta[probe], density_beta, -hybrid, None, probe),
                )
            )
            if omega != 0:
                long_range = -(alpha - hybrid)
                k_terms.extend(
                    (
                        common.JKDerivativeTerm(p_alpha[probe], density_alpha, long_range, omega, probe),
                        common.JKDerivativeTerm(p_beta[probe], density_beta, long_range, omega, probe),
                    )
                )
    xctype = ni._xc_type(mf.xc)
    if xctype != 'HF':
        if xctype == 'LDA':
            derivative_contractor = xc.contract_lda_vxc_derivative
        elif xctype == 'GGA':
            derivative_contractor = xc.contract_gga_vxc_derivative
        elif xctype == 'MGGA':
            derivative_contractor = xc.contract_mgga_vxc_derivative
        else:
            raise NotImplementedError('ordinary Fock direct derivative is not implemented for %s' % xctype)
        if tdobj.nobeta and not isinstance(mf, (AverageOccupationROKS, SymAdaptedAverageOccupationROKS)):
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
    return gradient, j_terms, k_terms


def _orbital_gradient(tdobj, m_matrix, adjoint, fock_contraction, atmlst=None):
    """Combine the Z-vector Fock derivative with the MO overlap response."""
    mol = tdobj.mol
    mf = tdobj._scf
    if atmlst is None:
        atmlst = range(mol.natm)
    atmlst = tuple(atmlst)
    mo = np.asarray(mf.mo_coeff)
    overlap_derivative = rhf_grad.Gradients(mf).get_ovlp(mol)
    offsets = mol.offset_nr_by_atom()
    result = np.zeros((len(atmlst), 3))
    for k, atom in enumerate(atmlst):
        p0, p1 = offsets[atom][2:]
        for xyz in range(3):
            overlap = np.zeros((mol.nao_nr(), mol.nao_nr()))
            overlap[p0:p1] += overlap_derivative[xyz, p0:p1]
            overlap[:, p0:p1] += overlap_derivative[xyz, p0:p1].T
            symmetric_kappa = -0.5 * (mo.conj().T @ overlap @ mo)
            result[k, xyz] = (
                -fock_contraction[k, xyz] - np.trace(adjoint.T @ symmetric_kappa) + np.trace(m_matrix @ symmetric_kappa)
            )
    return result
