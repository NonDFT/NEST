"""Analytic gradient for current NTTDA ``deltaS=0``.

The public ``grad_elec`` function exposes the complete scientific data flow.
All same-spin amplitudes, Fock projections, response coefficients, and direct
J/K contractions live in this module.  Only XC quadrature and the ROKS
adjoint are delegated to sibling modules.
"""

from dataclasses import dataclass

import numpy as np

from pyscf import dft, lib
from nest.nttda import nttda as nttda_mod
from nest.nttda.nttda import gen_rohf_response_sc

from . import xc as xc_backend
from .roks import finish_gradient



# Orbital spaces and native amplitudes

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


@dataclass(frozen=True)
class SameSpinAmplitudes:
    """Five amplitude blocks used by ``NTTDA(deltaS=0)``."""

    co: np.ndarray
    cv: np.ndarray
    oo: float
    ov: np.ndarray
    cv0: np.ndarray


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


def pair_density(c_left, coefficient, c_right):
    """Build ``C_left coefficient C_right^T`` without symmetrizing it."""
    return c_left @ np.asarray(coefficient) @ c_right.conj().T


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
    if tdobj.nobeta:
        dm_alpha, dm_beta = mf.make_rdm1()
        dm0 = 0.5 * (dm_alpha + dm_beta)
        fock = mf.get_fock(dm=np.asarray((dm0, dm0)))
    else:
        fock = mf.get_fock()
    fock0 = 0.5 * (fock.focka + fock.fockb)
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


def _fock_response_q(tdobj, p_alpha, p_beta):
    """Reference-density derivative of a spin-resolved Fock scalar."""
    mf = tdobj._scf
    mo = np.asarray(mf.mo_coeff)
    occ_alpha = np.flatnonzero(mf.mo_occ > 0)
    occ_beta = np.flatnonzero(mf.mo_occ == 2)
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
    q_alpha = np.zeros((mo.shape[1], mo.shape[1]))
    q_beta = np.zeros_like(q_alpha)
    q_alpha[:, occ_alpha] = (
        mo.conj().T @ (v_alpha + v_alpha.T) @ mo[:, occ_alpha]
    )
    q_beta[:, occ_beta] = (
        mo.conj().T @ (v_beta + v_beta.T) @ mo[:, occ_beta]
    )
    return q_alpha, q_beta


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

@dataclass(frozen=True)
class ResponseTerm:
    """Directed response term from one source density to one target block."""

    target: str
    source: str
    vref0: float
    vref1: float


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
            vref1 = ni.nr_rks_fxc(
                mol, mf.grids, mf.xc, None, dms, 0, 0,
                None, None, fxc_ref, max_memory=max_memory,
            )
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
                "NTTDA same-spin response does not support XC type %s" % xctype
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


# AO J/K nuclear derivatives

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
    """Channel-local scheduler for fixed-AO J/K derivative contractions."""

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


def _reference_spin_densities(tdobj):
    mf = tdobj._scf
    mo = np.asarray(mf.mo_coeff)
    return (
        mo[:, mf.mo_occ > 0] @ mo[:, mf.mo_occ > 0].T,
        mo[:, mf.mo_occ == 2] @ mo[:, mf.mo_occ == 2].T,
    )


def _spin_probe_stacks(p_alpha, p_beta):
    p_alpha = np.asarray(p_alpha)
    p_beta = np.asarray(p_beta)
    single_probe = p_alpha.ndim == 2
    if single_probe:
        p_alpha = p_alpha[None]
        p_beta = p_beta[None]
    return p_alpha, p_beta, single_probe


def spin_fock_direct_dft(
        gradient_driver, tdobj, p_alpha, p_beta, atmlst=None,
        nobeta_p0=None, jk_ledger=None, output_slots=None):
    """Differentiate one or more ordinary UKS Fock scalar probes.

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
    hcore_derivative = mf.nuc_grad_method().hcore_generator(mol)
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
    if nobeta_p0 is not None and tdobj.nobeta:
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


def spin_fock_direct_hf(
        gradient_driver, tdobj, p_alpha, p_beta, atmlst=None,
        jk_ledger=None, output_slots=None):
    """Differentiate one or more spin-resolved HF Fock scalar probes."""
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
    dm_alpha, dm_beta = _reference_spin_densities(tdobj)
    gradient = np.zeros((len(p_alpha), len(atmlst), 3))

    hcore_derivative = tdobj._scf.nuc_grad_method().hcore_generator(mol)
    for k, atom in enumerate(atmlst):
        gradient[:, k] += lib.einsum(
            "npq,xpq->nx", p_total, hcore_derivative(atom),
        )
    local_ledger = _JKDerivativeLedger()
    ledger = jk_ledger if jk_ledger is not None else local_ledger
    for probe in range(len(p_alpha)):
        ledger.add(
            "j", output_slots[probe], (
                (p_total[probe], dm_alpha, 1.0, None),
                (p_total[probe], dm_beta, 1.0, None),
            ),
        )
        ledger.add(
            "k", output_slots[probe], (
                (p_alpha[probe], dm_alpha, -1.0, None),
                (p_beta[probe], dm_beta, -1.0, None),
            ),
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


# Hybrid/RSH Fz correction

def same_spin_fockz_hfx_terms(
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
    """Build the complete analytic excitation gradient for deltaS=0.

    The function follows the physical order of the Lagrangian: native
    amplitudes and AO probes, XC/J/K contributions to the M matrix and direct
    derivative, the ROKS adjoint, and the final overlap contraction.
    """
    if tdobj.deltaS != 0:
        raise ValueError("deltaS=0 gradient received a different spin channel")
    if atmlst is None:
        atmlst = range(tdobj.mol.natm)
    atmlst = tuple(atmlst)
    mf = tdobj._scf
    xctype = mf._numint._xc_type(mf.xc)

    # 1. Native amplitudes, transition densities, and explicit Fock probes.
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
    jk_ledger = _JKDerivativeLedger()
    direct_slot = "direct"
    zvector_slot = "zvector"

    # 2. Explicit Fock contribution to the orbital-rotation M matrix.
    fock_alpha, fock_beta = same_spin_fock_q(tdobj, xy)

    if xctype == "HF":
        # 3a. HF response and fixed-orbital AO derivative.
        response_alpha, response_beta = same_spin_response_projection_q(
            tdobj, xy,
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
        # 3b. Semilocal XC, hybrid/RSH, Fz, and nobeta contributions.
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
                "NTTDA deltaS=0 gradient does not support XC type %s" % xctype
            ) from error

        hfx_alpha, hfx_beta = same_spin_hfx_projection_q(tdobj, xy)
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
        fockz_hfx = same_spin_fockz_hfx_terms(
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

    # 4-5. ROKS transpose-Hessian adjoint, Dz Fock derivative, and Pulay term.
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
