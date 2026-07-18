"""Analytic gradient for current NTTDA ``deltaS=+1``.

The spin-raising channel contains one closed-to-virtual amplitude block.  Its
complete scalar is

``P0:F0 + Pz:Fz + T:K_perp[T]``

with ``P0 = D_virtual - D_closed`` and
``Pz = D_virtual + D_closed``.  This module differentiates that scalar while
reusing only the channel-independent AO J/K and ROKS-adjoint infrastructure.
"""

from dataclasses import dataclass

import numpy as np

from pyscf import dft, lib
from nest.nttda.nttda import gen_rohf_response_sfu

from . import xc as xc_backend
from .delta_s_minus_one import (
    _JKDerivativeLedger,
    response_direct_hfx,
    spin_fock_direct_dft,
    spin_fock_direct_hf,
    spin_lowering_fockz_hfx_terms as spin_raising_fockz_hfx_terms,
)
from .roks import finish_gradient


@dataclass(frozen=True)
class OrbitalSpaces:
    """Closed, open, and virtual spatial-orbital partitions."""

    closed: np.ndarray
    open: np.ndarray
    virtual: np.ndarray
    c_closed: np.ndarray
    c_open: np.ndarray
    c_virtual: np.ndarray


@dataclass(frozen=True)
class SpinRaisingAmplitudes:
    """The sole closed-to-virtual ``deltaS=+1`` amplitude block."""

    cv: np.ndarray


@dataclass(frozen=True)
class FockProjection:
    """One scalar term ``P:(weight_f0 F0 + weight_fz Fz)``."""

    name: str
    indices: np.ndarray
    orbitals: np.ndarray
    coefficient: np.ndarray
    weight_f0: float
    weight_fz: float

    def density(self):
        return self.orbitals @ self.coefficient @ self.orbitals.conj().T


@dataclass(frozen=True)
class ResponseTerm:
    """One directed transverse-response scalar term."""

    target: str
    source: str
    vref0: float
    vref1: float


def orbital_spaces(tdobj):
    """Return the ROKS ``C/O/V`` partition used by spin raising."""
    mf = tdobj._scf
    occupation = np.asarray(mf.mo_occ)
    if occupation.ndim != 1:
        raise ValueError("NTTDA gradients require spatial ROKS orbitals")
    closed = np.flatnonzero(occupation == 2)
    open_ = np.flatnonzero(occupation == 1)
    virtual = np.flatnonzero(occupation == 0)
    coefficient = np.asarray(mf.mo_coeff)
    return OrbitalSpaces(
        closed=closed,
        open=open_,
        virtual=virtual,
        c_closed=coefficient[:, closed],
        c_open=coefficient[:, open_],
        c_virtual=coefficient[:, virtual],
    )


def split_spin_raising(tdobj, xy):
    """Validate and return the native ``C->V`` spin-raising amplitude."""
    spaces = orbital_spaces(tdobj)
    vector = xy[0] if isinstance(xy, (tuple, list)) else xy
    vector = np.asarray(vector)
    expected = (len(spaces.closed), len(spaces.virtual))
    if vector.size != expected[0] * expected[1]:
        raise ValueError(
            "deltaS=+1 amplitude has size %d; expected %d" %
            (vector.size, expected[0] * expected[1])
        )
    return spaces, SpinRaisingAmplitudes(vector.reshape(expected))


def pair_density(c_left, coefficient, c_right):
    """Build ``C_left coefficient C_right.T`` without symmetrizing it."""
    return c_left @ np.asarray(coefficient) @ c_right.conj().T


def spin_raising_transition_densities(tdobj, xy):
    """Return the alpha-target/beta-source directed transition density."""
    spaces, amplitudes = split_spin_raising(tdobj, xy)
    return spaces, amplitudes, {
        "CV": pair_density(
            spaces.c_virtual, amplitudes.cv.T, spaces.c_closed,
        ),
    }


def spin_raising_block_data(spaces, amplitudes):
    """MO indices and coefficients for the sole directed density."""
    return {
        "CV": (spaces.virtual, spaces.closed, amplitudes.cv.T),
    }


def spin_raising_response_terms():
    """The single ``CV-CV`` transverse-kernel contraction."""
    return (ResponseTerm("CV", "CV", 1.0, 0.0),)


def spin_raising_fock0_fockz(tdobj, max_memory=None):
    """Build exactly the operators used by ``gen_vind_sfu``."""
    mf = tdobj._scf
    if max_memory is None:
        max_memory = tdobj.max_memory
    _response, fockz = gen_rohf_response_sfu(
        mf,
        mo_coeff=mf.mo_coeff,
        mo_occ=mf.mo_occ,
        hermi=0,
        max_memory=max_memory,
    )
    if tdobj.nobeta:
        density_alpha, density_beta = mf.make_rdm1()
        density0 = 0.5 * (density_alpha + density_beta)
        fock = mf.get_fock(dm=np.asarray((density0, density0)))
    else:
        fock = mf.get_fock()
    fock0 = 0.5 * (fock.focka + fock.fockb)
    return fock0, fockz


def spin_raising_fock_projections(tdobj, xy):
    """Return the two-term explicit-Fock ledger."""
    spaces, amplitudes = split_spin_raising(tdobj, xy)
    x = amplitudes.cv
    return (
        FockProjection(
            name="particle-vv",
            indices=spaces.virtual,
            orbitals=spaces.c_virtual,
            coefficient=x.T @ x,
            weight_f0=1.0,
            weight_fz=1.0,
        ),
        FockProjection(
            name="hole-cc",
            indices=spaces.closed,
            orbitals=spaces.c_closed,
            coefficient=-(x @ x.T),
            weight_f0=1.0,
            weight_fz=-1.0,
        ),
    )


def spin_raising_fock_probes(tdobj, xy):
    """Return ``P0=Dv-Dc`` and ``Pz=Dv+Dc`` AO probes."""
    nao = tdobj.mol.nao_nr()
    p0 = np.zeros((nao, nao))
    pz = np.zeros_like(p0)
    for term in spin_raising_fock_projections(tdobj, xy):
        density = term.density()
        p0 += term.weight_f0 * density
        pz += term.weight_fz * density
    return p0, pz


def spin_raising_fock_scalar(tdobj, xy, max_memory=None):
    """Evaluate ``P0:F0 + Pz:Fz``."""
    fock0, fockz = spin_raising_fock0_fockz(
        tdobj, max_memory=max_memory,
    )
    p0, pz = spin_raising_fock_probes(tdobj, xy)
    return float(
        lib.einsum("pq,pq->", p0, fock0)
        + lib.einsum("pq,pq->", pz, fockz)
    )


def _transverse_potential(
        tdobj, density, max_memory=None, hfx_only=False):
    """Apply the full or exact-exchange-only transverse kernel."""
    mf = tdobj._scf
    density = np.asarray(density)
    if max_memory is None:
        max_memory = tdobj.max_memory
    if not hfx_only:
        response, _fockz = gen_rohf_response_sfu(
            mf,
            mo_coeff=mf.mo_coeff,
            mo_occ=mf.mo_occ,
            hermi=0,
            max_memory=max_memory,
        )
        return response(density[None])[0]

    potential = np.zeros_like(density)
    ni = mf._numint
    if not ni.libxc.is_hybrid_xc(mf.xc):
        return potential
    omega, alpha, hybrid = ni.rsh_and_hybrid_coeff(
        mf.xc, mf.mol.spin,
    )
    potential -= hybrid * mf.get_k(mf.mol, density, hermi=0)
    if omega != 0:
        potential -= (alpha - hybrid) * mf.get_k(
            mf.mol, density, hermi=0, omega=omega,
        )
    return potential


def spin_raising_response_scalar(tdobj, xy, max_memory=None):
    """Evaluate the sole ``T:K_perp[T]`` response scalar."""
    _spaces, _amplitudes, densities = spin_raising_transition_densities(
        tdobj, xy,
    )
    density = densities["CV"]
    potential = _transverse_potential(
        tdobj, density, max_memory=max_memory,
    )
    return float(lib.einsum("pq,pq->", density, potential))


def spin_raising_ledger_scalar(tdobj, xy, max_memory=None):
    """Independent reconstruction of ``X.T gen_vind_sfu(X)``."""
    return (
        spin_raising_fock_scalar(tdobj, xy, max_memory=max_memory)
        + spin_raising_response_scalar(tdobj, xy, max_memory=max_memory)
    )


def spin_raising_action_scalar(tdobj, xy):
    """Evaluate ``X.T gen_vind_sfu(X)`` with the production action."""
    vector = xy[0] if isinstance(xy, (tuple, list)) else xy
    vector = np.asarray(vector)
    action, _diagonal = tdobj.gen_vind_sfu()
    result = action(vector.reshape(1, -1)).reshape(vector.shape)
    return float(np.vdot(vector, result).real)


def _fock_response_q(tdobj, p_alpha, p_beta):
    """Reference-density derivative of a spin-resolved Fock scalar."""
    mf = tdobj._scf
    mo = np.asarray(mf.mo_coeff)
    occupied_alpha = np.flatnonzero(mf.mo_occ > 0)
    occupied_beta = np.flatnonzero(mf.mo_occ == 2)
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
        v_alpha = coulomb - mf.get_k(
            mf.mol, p_alpha.T, hermi=0,
        )
        v_beta = coulomb - mf.get_k(
            mf.mol, p_beta.T, hermi=0,
        )
    q_alpha = np.zeros((mo.shape[1], mo.shape[1]))
    q_beta = np.zeros_like(q_alpha)
    q_alpha[:, occupied_alpha] = (
        mo.conj().T @ (v_alpha + v_alpha.T) @ mo[:, occupied_alpha]
    )
    q_beta[:, occupied_beta] = (
        mo.conj().T @ (v_beta + v_beta.T) @ mo[:, occupied_beta]
    )
    return q_alpha, q_beta


def spin_raising_fock_q(tdobj, xy, max_memory=None):
    """Explicit-Fock projection and reference-density response M matrices."""
    mf = tdobj._scf
    mo = np.asarray(mf.mo_coeff)
    nmo = mo.shape[1]
    fock0, fockz = spin_raising_fock0_fockz(
        tdobj, max_memory=max_memory,
    )
    fock0_mo = mo.conj().T @ fock0 @ mo
    fockz_mo = mo.conj().T @ fockz @ mo
    q_alpha = np.zeros((nmo, nmo))
    q_beta = np.zeros_like(q_alpha)
    is_hf = mf._numint._xc_type(mf.xc) == "HF"

    for term in spin_raising_fock_projections(tdobj, xy):
        indices = term.indices
        coefficient = term.coefficient

        def project(target, operator, scale):
            if scale:
                target[:, indices] += (
                    scale * operator[:, indices] @ coefficient.T
                )
                target[:, indices] += (
                    scale * operator[:, indices] @ coefficient
                )

        project(q_alpha, fock0_mo, 0.5 * term.weight_f0)
        project(q_beta, fock0_mo, 0.5 * term.weight_f0)
        if is_hf:
            project(q_alpha, fockz_mo, 0.5 * term.weight_fz)
            project(q_beta, fockz_mo, 0.5 * term.weight_fz)
        else:
            project(q_alpha, fockz_mo, term.weight_fz)

    p0, pz = spin_raising_fock_probes(tdobj, xy)
    p_alpha = 0.5 * p0
    p_beta = 0.5 * p0
    if is_hf:
        p_alpha = p_alpha + 0.5 * pz
        p_beta = p_beta - 0.5 * pz
    response_alpha, response_beta = _fock_response_q(
        tdobj, p_alpha, p_beta,
    )
    q_alpha += response_alpha
    q_beta += response_beta
    return q_alpha, q_beta


def spin_raising_response_projection_q(
        tdobj, xy, max_memory=None, hfx_only=False):
    """Transition-factor derivative of ``T:K_perp[T]``."""
    spaces, amplitudes, densities = spin_raising_transition_densities(
        tdobj, xy,
    )
    potential = 2.0 * _transverse_potential(
        tdobj,
        densities["CV"],
        max_memory=max_memory,
        hfx_only=hfx_only,
    )
    potential_mo = (
        np.asarray(tdobj._scf.mo_coeff).conj().T
        @ potential @ np.asarray(tdobj._scf.mo_coeff)
    )
    nmo = potential_mo.shape[0]
    q_alpha = np.zeros((nmo, nmo))
    q_beta = np.zeros_like(q_alpha)
    q_alpha[:, spaces.virtual] += (
        potential_mo[:, spaces.closed] @ amplitudes.cv
    )
    q_beta[:, spaces.closed] += (
        potential_mo[spaces.virtual, :].T @ amplitudes.cv.T
    )
    return q_alpha, q_beta


def grad_elec(
        gradient_driver, tdobj, xy, atmlst=None, tolerance=1e-12,
        max_cycle=None):
    """Build the complete analytic excitation gradient for deltaS=+1."""
    if tdobj.deltaS != 1:
        raise ValueError("deltaS=+1 gradient received a different spin channel")
    if atmlst is None:
        atmlst = range(tdobj.mol.natm)
    atmlst = tuple(atmlst)
    mf = tdobj._scf
    xctype = mf._numint._xc_type(mf.xc)

    spaces, amplitudes, densities = spin_raising_transition_densities(
        tdobj, xy,
    )
    blocks = spin_raising_block_data(spaces, amplitudes)
    response_terms = spin_raising_response_terms()
    channel_data = (
        spaces, amplitudes, densities, blocks, response_terms, "alpha",
    )
    p0, pz = spin_raising_fock_probes(tdobj, xy)
    jk_ledger = _JKDerivativeLedger()
    direct_slot = "direct"
    zvector_slot = "zvector"

    fock_alpha, fock_beta = spin_raising_fock_q(tdobj, xy)

    if xctype == "HF":
        response_alpha, response_beta = (
            spin_raising_response_projection_q(tdobj, xy)
        )
        m_matrix = (
            fock_alpha + fock_beta + response_alpha + response_beta
        )
        direct_fock_probes = (
            0.5 * (p0 + pz),
            0.5 * (p0 - pz),
        )
        direct = response_direct_hfx(
            gradient_driver,
            tdobj,
            densities,
            response_terms,
            atmlst=atmlst,
            jk_ledger=jk_ledger,
            output_slot=direct_slot,
        )

        def fock_direct(driver, obj, p_alpha, p_beta, atmlst=None):
            local = spin_fock_direct_hf(
                driver,
                obj,
                p_alpha,
                p_beta,
                atmlst=atmlst,
                jk_ledger=jk_ledger,
                output_slots=(direct_slot, zvector_slot),
            )
            contractions = jk_ledger.contract(
                driver,
                obj.mol,
                atmlst,
                slots=(direct_slot, zvector_slot),
            )
            local[0] += contractions[direct_slot]
            local[1] += contractions[zvector_slot]
            return local
    else:
        try:
            response_builder, fockz_builder, nobeta_q_builder = {
                "LDA": (
                    xc_backend.lda_response_terms,
                    xc_backend.lda_fockz_terms,
                    xc_backend.lda_nobeta_reference_q,
                ),
                "GGA": (
                    xc_backend.gga_response_terms,
                    xc_backend.gga_fockz_terms,
                    xc_backend.gga_nobeta_reference_q,
                ),
                "MGGA": (
                    xc_backend.mgga_response_terms,
                    xc_backend.mgga_fockz_terms,
                    xc_backend.mgga_nobeta_reference_q,
                ),
            }[xctype]
        except KeyError as error:
            raise NotImplementedError(
                "NTTDA deltaS=+1 gradient does not support XC type %s" %
                xctype
            ) from error

        hfx_alpha, hfx_beta = spin_raising_response_projection_q(
            tdobj, xy, hfx_only=True,
        )
        response_xc = response_builder(
            gradient_driver,
            tdobj,
            channel_data,
            atmlst=atmlst,
        )
        fockz_xc = fockz_builder(
            gradient_driver,
            tdobj,
            spaces,
            pz,
            atmlst=atmlst,
        )
        fockz_hfx = spin_raising_fockz_hfx_terms(
            gradient_driver,
            tdobj,
            pz,
            atmlst=atmlst,
            jk_ledger=jk_ledger,
            output_slot=direct_slot,
        )
        common_alpha, common_beta = nobeta_q_builder(tdobj, p0)
        m_matrix = (
            fock_alpha + fock_beta
            + hfx_alpha + hfx_beta
            + response_xc.q_alpha + response_xc.q_beta
            + fockz_xc.q_alpha + fockz_xc.q_beta
            + fockz_hfx.q_alpha + fockz_hfx.q_beta
            + common_alpha + common_beta
        )

        direct_fock_probes = (0.5 * p0, 0.5 * p0)
        direct = response_direct_hfx(
            gradient_driver,
            tdobj,
            densities,
            response_terms,
            atmlst=atmlst,
            jk_ledger=jk_ledger,
            output_slot=direct_slot,
        )
        direct += response_xc.direct
        direct += fockz_xc.direct
        direct += fockz_hfx.direct

        def fock_direct(driver, obj, p_alpha, p_beta, atmlst=None):
            local = spin_fock_direct_dft(
                driver,
                obj,
                p_alpha,
                p_beta,
                atmlst=atmlst,
                nobeta_p0=p0,
                jk_ledger=jk_ledger,
                output_slots=(direct_slot, zvector_slot),
            )
            contractions = jk_ledger.contract(
                driver,
                obj.mol,
                atmlst,
                slots=(direct_slot, zvector_slot),
            )
            local[0] += contractions[direct_slot]
            local[1] += contractions[zvector_slot]
            return local

    return finish_gradient(
        gradient_driver,
        tdobj,
        m_matrix,
        direct,
        atmlst,
        tolerance,
        max_cycle,
        fock_direct,
        direct_fock_probes=direct_fock_probes,
    )


__all__ = [
    "FockProjection",
    "OrbitalSpaces",
    "ResponseTerm",
    "SpinRaisingAmplitudes",
    "grad_elec",
    "orbital_spaces",
    "spin_raising_action_scalar",
    "spin_raising_fock_probes",
    "spin_raising_ledger_scalar",
    "spin_raising_transition_densities",
    "split_spin_raising",
]
