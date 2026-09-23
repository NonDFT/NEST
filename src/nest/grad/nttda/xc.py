"""LDA, GGA, and meta-GGA quadrature for NTTDA gradients.

This module is channel-neutral: callers provide orbital spaces, transition
densities, block projections, and response-term coefficients.
"""

from dataclasses import dataclass

import numpy as np

from pyscf import lib
from pyscf.dft.gen_grid import NBINS
from pyscf.dft.numint import _dot_ao_ao_sparse, _scale_ao_sparse
from pyscf.grad import tdrks as tdrks_grad


# Shared result and projection helpers

@dataclass(frozen=True)
class XCGradientTerms:
    q_alpha: np.ndarray
    q_beta: np.ndarray
    direct: np.ndarray


# AO feature algebra

def sparse_context(mf):
    cutoff = mf.grids.cutoff * 1e2
    nbins = NBINS * 2 - int(NBINS * np.log(cutoff) / np.log(mf.grids.cutoff))
    pair_mask = mf.mol.get_overlap_cond() < -np.log(mf._numint.cutoff)
    return nbins, pair_mask, mf.mol.ao_loc_nr()


def add_gga_matrix(mol, output, ao, weights, mask, sparse):
    nbins, pair_mask, ao_loc = sparse
    weights = np.asarray(weights, order="C").copy()
    weights[0] *= 0.5
    scaled = _scale_ao_sparse(ao[:4], weights, mask, ao_loc)
    matrix = _dot_ao_ao_sparse(
        ao[0], scaled, None, nbins, mask, pair_mask, ao_loc,
        hermi=0, out=None,
    )
    output += lib.hermi_sum(matrix)


def add_mgga_matrix(mol, output, ao, weights, mask, sparse=None):
    """Accumulate one ordinary meta-GGA feature potential matrix."""
    del sparse
    output += mgga_eval_matrix(mol, ao, weights, mask)


def pair_matrix(mol, ao, mask, tensor, sparse):
    nbins, pair_mask, ao_loc = sparse
    output = np.zeros((mol.nao_nr(), mol.nao_nr()))
    for left in range(4):
        scaled = _scale_ao_sparse(
            ao[:4], np.asarray(tensor[left], order="C"), mask, ao_loc,
        )
        output += _dot_ao_ao_sparse(
            ao[left], scaled, None, nbins, mask, pair_mask, ao_loc,
            hermi=0, out=None,
        )
    return output


def second_derivative_index(first, second):
    if first > second:
        first, second = second, first
    return {
        (0, 0): 4,
        (0, 1): 5,
        (0, 2): 6,
        (1, 1): 7,
        (1, 2): 8,
        (2, 2): 9,
    }[(first, second)]


def _compact_ao_center_derivative(ao, p0, p1, xyz, xctype):
    """AO-center derivative restricted to one atom's AO columns."""
    if xctype == "LDA":
        return -ao[xyz + 1][:, p0:p1]
    delta = np.empty((4, ao.shape[-2], p1 - p0))
    delta[0] = -ao[xyz + 1][:, p0:p1]
    for feature in range(3):
        delta[feature + 1] = -ao[
            second_derivative_index(xyz, feature)
        ][:, p0:p1]
    return delta


def _hermitian_density_derivative_batches(
        ao, densities, p0, p1, xctype):
    """Yield AO-center derivatives for a stack of real symmetric densities."""
    densities = np.asarray(densities)
    feature_count = 1 if xctype == "LDA" else 4
    density_rows = densities[:, p0:p1]
    packed_rows = density_rows.transpose(2, 0, 1).reshape(
        density_rows.shape[-1], -1,
    )
    contracted = (ao[:feature_count] @ packed_rows).reshape(
        feature_count, ao.shape[-2], len(densities), p1 - p0,
    ).transpose(2, 0, 1, 3)

    for xyz in range(3):
        delta = _compact_ao_center_derivative(
            ao, p0, p1, xyz, xctype,
        )
        if xctype == "LDA":
            yield 2.0 * lib.einsum(
                "ga,nga->ng", delta, contracted[:, 0],
            )[:, None]
            continue

        derivative_count = 4 if xctype == "GGA" else 5
        output = np.empty((
            len(densities), derivative_count, ao.shape[-2],
        ))
        output[:, 0] = 2.0 * lib.einsum(
            "ga,nga->ng", delta[0], contracted[:, 0],
        )
        for feature in range(1, 4):
            output[:, feature] = 2.0 * (
                lib.einsum(
                    "ga,nga->ng", delta[feature], contracted[:, 0],
                )
                + lib.einsum(
                    "ga,nga->ng", delta[0], contracted[:, feature],
                )
            )
        if xctype == "GGA":
            yield output
            continue
        output[:, 4] = 0.0
        for feature in range(1, 4):
            output[:, 4] += lib.einsum(
                "ga,nga->ng", delta[feature], contracted[:, feature],
            )
        yield output


def pair_feature_batches(ao, densities):
    """Pair features and reusable ``AO @ D`` contractions by channel."""
    densities = np.asarray(densities)
    grids = ao.shape[-2]
    features = np.empty((len(densities), 4, 4, grids))
    contracted = np.asarray([
        ao[index] @ densities for index in range(4)
    ]).transpose(1, 0, 2, 3)
    for left in range(4):
        for right in range(4):
            features[:, left, right] = lib.einsum(
                "ngu,gu->ng", contracted[:, left], ao[right],
            )
    return features, contracted


def contract_pair_feature_derivatives(
        ao, densities, delta, contracted_ao, p0, p1,
        tensor_weights, grid_weights):
    """Contract pair-feature derivatives without materializing ``dPair``."""
    densities = np.asarray(densities)
    tensor_weights = np.asarray(tensor_weights)
    value = 0.0
    for left in range(4):
        contracted_delta = delta[left] @ densities[:, p0:p1]
        value += lib.einsum(
            "pbg,pgu,bgu,g->",
            tensor_weights[:, left], contracted_delta, ao[:4],
            grid_weights, optimize=True,
        )
    contracted_atom = contracted_ao[:, :, :, p0:p1]
    for right in range(4):
        value += lib.einsum(
            "pag,pagq,gq,g->",
            tensor_weights[:, :, right], contracted_atom,
            delta[right], grid_weights, optimize=True,
        )
    return value


def gga_pair_potential(kernel, features):
    output = np.zeros_like(features)
    output[0, 0] = lib.einsum("abg,abg->g", kernel, features)
    output[1:4, 0] = kernel[1:4, 0] * features[0, 0]
    output[1:4, 0] += lib.einsum(
        "ijg,jg->ig", kernel[1:4, 1:4], features[0, 1:4],
    )
    output[0, 1:4] = kernel[0, 1:4] * features[0, 0]
    output[0, 1:4] += lib.einsum(
        "ijg,ig->jg", kernel[1:4, 1:4], features[1:4, 0],
    )
    output[1:4, 1:4] = kernel[1:4, 1:4] * features[0, 0]
    return output


def gga_pair_kernel_cross(left, right):
    output = np.zeros_like(left)
    output[0, 0] = left[0, 0] * right[0, 0]
    output[1:4, 0] = (
        left[0, 0][None] * right[1:4, 0]
        + left[1:4, 0] * right[0, 0][None]
    )
    output[0, 1:4] = (
        left[0, 0][None] * right[0, 1:4]
        + left[0, 1:4] * right[0, 0][None]
    )
    output[1:4, 1:4] = (
        left[0, 0][None, None] * right[1:4, 1:4]
        + left[1:4, 0][:, None] * right[0, 1:4][None]
        + left[0, 1:4][None] * right[1:4, 0][:, None]
        + left[1:4, 1:4] * right[0, 0][None, None]
    )
    return output


def mgga_pair_potential(kernel, features):
    output = np.zeros_like(features)
    output[0, 0] = lib.einsum(
        "abg,abg->g", kernel[:4, :4], features,
    )
    output[1:4, 0] = kernel[1:4, 0] * features[0, 0]
    output[1:4, 0] += lib.einsum(
        "ijg,jg->ig", kernel[1:4, 1:4], features[0, 1:4],
    )
    output[1:4, 0] += 0.5 * kernel[4, 0][None] * features[1:4, 0]
    output[1:4, 0] += 0.5 * lib.einsum(
        "jg,ijg->ig", kernel[4, 1:4], features[1:4, 1:4],
    )
    output[0, 1:4] = kernel[0, 1:4] * features[0, 0]
    output[0, 1:4] += lib.einsum(
        "ijg,ig->jg", kernel[1:4, 1:4], features[1:4, 0],
    )
    output[0, 1:4] += 0.5 * kernel[0, 4][None] * features[0, 1:4]
    output[0, 1:4] += 0.5 * lib.einsum(
        "ig,ijg->jg", kernel[1:4, 4], features[1:4, 1:4],
    )
    output[1:4, 1:4] = kernel[1:4, 1:4] * features[0, 0]
    output[1:4, 1:4] += 0.5 * lib.einsum(
        "ig,jg->ijg", kernel[1:4, 4], features[0, 1:4],
    )
    output[1:4, 1:4] += 0.5 * lib.einsum(
        "jg,ig->ijg", kernel[4, 1:4], features[1:4, 0],
    )
    output[1:4, 1:4] += 0.25 * kernel[4, 4][None, None] * features[1:4, 1:4]
    return output


def mgga_pair_kernel_cross(left, right):
    grids = left.shape[-1]
    output = np.zeros((5, 5, grids))
    output[:4, :4] = gga_pair_kernel_cross(left, right)
    output[4, 0] = 0.5 * lib.einsum(
        "ig,ig->g", left[1:4, 0], right[1:4, 0],
    )
    output[0, 4] = 0.5 * lib.einsum(
        "jg,jg->g", left[0, 1:4], right[0, 1:4],
    )
    output[4, 1:4] = 0.5 * lib.einsum(
        "ig,ijg->jg", left[1:4, 0], right[1:4, 1:4],
    )
    output[4, 1:4] += 0.5 * lib.einsum(
        "ijg,ig->jg", left[1:4, 1:4], right[1:4, 0],
    )
    output[1:4, 4] = 0.5 * lib.einsum(
        "jg,ijg->ig", left[0, 1:4], right[1:4, 1:4],
    )
    output[1:4, 4] += 0.5 * lib.einsum(
        "ijg,jg->ig", left[1:4, 1:4], right[0, 1:4],
    )
    output[4, 4] = 0.25 * lib.einsum(
        "ijg,ijg->g", left[1:4, 1:4], right[1:4, 1:4],
    )
    return output


def gga_eval_matrix(mol, ao, weights, mask):
    output = np.zeros((4, mol.nao_nr(), mol.nao_nr()))
    tdrks_grad._gga_eval_mat_(
        mol, output, ao, np.array(weights, copy=True), mask,
        (0, mol.nbas), mol.ao_loc_nr(),
    )
    return output[0]


def mgga_eval_matrix(mol, ao, weights, mask):
    output = np.zeros((4, mol.nao_nr(), mol.nao_nr()))
    tdrks_grad._mgga_eval_mat_(
        mol, output, ao, np.array(weights, copy=True), mask,
        (0, mol.nbas), mol.ao_loc_nr(),
    )
    return output[0]


def _lda_matrix(ao0, weights):
    return ao0.T @ (ao0 * np.asarray(weights)[:, None])


def _project_channel_potentials(tdobj, potentials, blocks):
    """Project transition-factor potentials for any NTTDA spin channel."""
    mo = np.asarray(tdobj._scf.mo_coeff)
    q_alpha = np.zeros((mo.shape[1], mo.shape[1]))
    q_beta = np.zeros_like(q_alpha)
    for label, (target, source, coefficient) in blocks.items():
        potential = mo.conj().T @ potentials[label] @ mo
        q_beta[:, target] += potential[:, source] @ coefficient.T
        q_alpha[:, source] += potential[target, :].T @ coefficient
    return q_alpha, q_beta


def _reference_spin_densities(tdobj):
    """Spin densities of the variational reference used by the XC kernel."""
    mf = tdobj._scf
    if getattr(mf, "is_average_occupation_reference", False):
        return tuple(np.asarray(dm) for dm in mf.make_rdm1s())
    mo = np.asarray(mf.mo_coeff)
    return (
        mo[:, mf.mo_occ > 0] @ mo[:, mf.mo_occ > 0].T,
        mo[:, mf.mo_occ == 2] @ mo[:, mf.mo_occ == 2].T,
    )


def _reference_spin_occupations(tdobj):
    """Per-orbital alpha/beta occupations of the reference density."""
    mf = tdobj._scf
    occupation = np.asarray(mf.mo_occ)
    if getattr(mf, "is_average_occupation_reference", False):
        return 0.5 * occupation, 0.5 * occupation
    return (occupation > 0).astype(float), (occupation == 2).astype(float)


def _add_reference_q(tdobj, q_alpha, q_beta, matrix_alpha, matrix_beta):
    mf = tdobj._scf
    mo = np.asarray(mf.mo_coeff)
    occupation_alpha, occupation_beta = _reference_spin_occupations(tdobj)
    q_alpha += (
        mo.conj().T @ (matrix_alpha + matrix_alpha.T) @ mo
    ) * occupation_alpha[None, :]
    q_beta += (
        mo.conj().T @ (matrix_beta + matrix_beta.T) @ mo
    ) * occupation_beta[None, :]


def _spin_probe_stacks(probe_alpha, probe_beta):
    probe_alpha = np.asarray(probe_alpha)
    probe_beta = np.asarray(probe_beta)
    single_probe = probe_alpha.ndim == 2
    if single_probe:
        probe_alpha = probe_alpha[None]
        probe_beta = probe_beta[None]
    probe_alpha = 0.5 * (
        probe_alpha + probe_alpha.swapaxes(-1, -2)
    )
    probe_beta = 0.5 * (
        probe_beta + probe_beta.swapaxes(-1, -2)
    )
    return probe_alpha, probe_beta, single_probe


def _xc_density(ni, mol, ao, density, mask, xctype, hermi=1):
    ao_values = ao[0] if xctype == "LDA" else ao
    rho = ni.eval_rho(
        mol, ao_values, density, mask, xctype, hermi=hermi,
        with_lapl=False,
    )
    return rho[None] if rho.ndim == 1 else rho


def _xc_ao_center_derivative(ao, p0, p1, xyz, xctype):
    return _compact_ao_center_derivative(
        ao, p0, p1, xyz, xctype,
    )


def _xc_density_derivatives(
        ao, densities, p0, p1, xctype, ao_center_derivative):
    """AO-center derivatives for a stack of probe/reference densities."""
    densities = np.asarray(densities)
    if xctype == "LDA":
        delta0 = ao_center_derivative
        output = lib.einsum(
            "ga,nau,gu->ng",
            delta0,
            densities[:, p0:p1],
            ao[0],
        )
        output += lib.einsum(
            "gu,nua,ga->ng",
            ao[0],
            densities[:, :, p0:p1],
            delta0,
        )
        return output[:, None]

    delta = ao_center_derivative
    feature_count = 4 if xctype == "GGA" else 5
    output = np.empty(
        (len(densities), feature_count, ao.shape[-2]),
    )
    output[:, 0] = lib.einsum(
        "ga,nau,gu->ng",
        delta[0],
        densities[:, p0:p1],
        ao[0],
    )
    output[:, 0] += lib.einsum(
        "gu,nua,ga->ng",
        ao[0],
        densities[:, :, p0:p1],
        delta[0],
    )
    for feature in range(1, 4):
        output[:, feature] = lib.einsum(
            "ga,nau,gu->ng",
            delta[feature],
            densities[:, p0:p1],
            ao[0],
        )
        output[:, feature] += lib.einsum(
            "gu,nua,ga->ng",
            ao[feature],
            densities[:, :, p0:p1],
            delta[0],
        )
        output[:, feature] += lib.einsum(
            "ga,nau,gu->ng",
            delta[0],
            densities[:, p0:p1],
            ao[feature],
        )
        output[:, feature] += lib.einsum(
            "gu,nua,ga->ng",
            ao[0],
            densities[:, :, p0:p1],
            delta[feature],
        )
    if xctype == "GGA":
        return output[:, :4]

    output[:, 4] = 0.0
    for feature in range(1, 4):
        output[:, 4] += 0.5 * lib.einsum(
            "ga,nau,gu->ng",
            delta[feature],
            densities[:, p0:p1],
            ao[feature],
        )
        output[:, 4] += 0.5 * lib.einsum(
            "gu,nua,ga->ng",
            ao[feature],
            densities[:, :, p0:p1],
            delta[feature],
        )
    return output


def _response_density_stack(
        densities, density_alpha, density_beta):
    labels = tuple(densities)
    stack = np.asarray(
        [densities[label] for label in labels]
        + [density_alpha, density_beta]
    )
    return labels, stack


def _response_density_derivatives(
        ao, density_stack, labels, p0, p1, xyz, xctype):
    """Generate every channel/reference density derivative from one AO delta."""
    ao_center_derivative = _xc_ao_center_derivative(
        ao, p0, p1, xyz, xctype,
    )
    derivatives = _xc_density_derivatives(
        ao, density_stack, p0, p1, xctype, ao_center_derivative,
    )
    channel_count = len(labels)
    channel_derivatives = dict(zip(
        labels, derivatives[:channel_count],
    ))
    return (
        channel_derivatives,
        derivatives[channel_count],
        derivatives[channel_count + 1],
        ao_center_derivative,
    )


def _contract_vxc_derivative(
        mf, density_alpha, density_beta, probe_alpha, probe_beta,
        atmlst, xctype, max_memory):
    """Contract all fixed-grid XC potential derivatives in one grid pass."""
    mol = mf.mol
    ni = mf._numint
    if atmlst is None:
        atmlst = range(mol.natm)
    atmlst = tuple(atmlst)
    probe_alpha, probe_beta, single_probe = _spin_probe_stacks(
        probe_alpha, probe_beta,
    )
    output = np.zeros((len(probe_alpha), len(atmlst), 3))
    if not atmlst:
        return output[0] if single_probe else output

    density_alpha = 0.5 * (
        np.asarray(density_alpha) + np.asarray(density_alpha).T
    )
    density_beta = 0.5 * (
        np.asarray(density_beta) + np.asarray(density_beta).T
    )
    probe_densities = np.stack(
        (probe_alpha, probe_beta), axis=1,
    ).reshape(-1, *probe_alpha.shape[1:])
    density_stack = np.concatenate((
        np.asarray((density_alpha, density_beta)),
        probe_densities,
    ))
    offsets = mol.offset_nr_by_atom()
    ao_deriv = 1 if xctype == "LDA" else 2
    for ao, mask, weights, _coords in ni.block_loop(
            mol, mf.grids, mol.nao_nr(), ao_deriv,
            max_memory=max_memory):
        rho = np.asarray([
            _xc_density(ni, mol, ao, density, mask, xctype)
            for density in density_stack
        ])
        reference_rho = rho[:2]
        probe_rho = rho[2:].reshape(
            len(probe_alpha), 2, *rho.shape[1:],
        )
        vxc, fxc = ni.eval_xc_eff(
            mf.xc, reference_rho, deriv=2, xctype=xctype, spin=1,
        )[1:3]

        for k, atom in enumerate(atmlst):
            p0, p1 = offsets[atom][2:]
            for xyz in range(3):
                ao_center_derivative = _xc_ao_center_derivative(
                    ao, p0, p1, xyz, xctype,
                )
                density_derivative = _xc_density_derivatives(
                    ao, density_stack, p0, p1, xctype,
                    ao_center_derivative,
                )
                reference_derivative = density_derivative[:2]
                probe_derivative = density_derivative[2:].reshape(
                    len(probe_alpha), 2, *density_derivative.shape[1:],
                )
                output[:, k, xyz] += lib.einsum(
                    "nsxg,sxg,g->n",
                    probe_derivative,
                    vxc,
                    weights,
                )
                response_weights = lib.einsum(
                    "axg,axbyg,g->byg",
                    reference_derivative,
                    fxc,
                    weights,
                )
                output[:, k, xyz] += lib.einsum(
                    "nbyg,byg->n", probe_rho, response_weights,
                )
    return output[0] if single_probe else output


def contract_lda_vxc_derivative(
        mf, density_alpha, density_beta, probe_alpha, probe_beta,
        atmlst=None, max_memory=2000):
    """Contract all requested LDA XC potential nuclear derivatives."""
    return _contract_vxc_derivative(
        mf, density_alpha, density_beta, probe_alpha, probe_beta,
        atmlst, "LDA", max_memory,
    )


def contract_gga_vxc_derivative(
        mf, density_alpha, density_beta, probe_alpha, probe_beta,
        atmlst=None, max_memory=2000):
    """Contract all requested GGA XC potential nuclear derivatives."""
    return _contract_vxc_derivative(
        mf, density_alpha, density_beta, probe_alpha, probe_beta,
        atmlst, "GGA", max_memory,
    )


def contract_mgga_vxc_derivative(
        mf, density_alpha, density_beta, probe_alpha, probe_beta,
        atmlst=None, max_memory=2000):
    """Contract all requested MGGA XC potential nuclear derivatives."""
    return _contract_vxc_derivative(
        mf, density_alpha, density_beta, probe_alpha, probe_beta,
        atmlst, "MGGA", max_memory,
    )


# XC quadrature
def _reference_fref_kref(mf, rho0, xctype):
    fxc, kxc = mf._numint.eval_xc_eff(
        mf.xc, (rho0, rho0), deriv=3, xctype=xctype, spin=1,
    )[2:4]
    fref = 0.5 * (
        fxc[0, :, 0] - fxc[0, :, 1]
        - fxc[1, :, 0] + fxc[1, :, 1]
    )
    kref_alpha = 0.5 * (
        kxc[0, :, 0, :, 0] - kxc[0, :, 1, :, 0]
        - kxc[1, :, 0, :, 0] + kxc[1, :, 1, :, 0]
    )
    kref_beta = 0.5 * (
        kxc[0, :, 0, :, 1] - kxc[0, :, 1, :, 1]
        - kxc[1, :, 0, :, 1] + kxc[1, :, 1, :, 1]
    )
    return fref, kref_alpha, kref_beta


def response_terms(
        gradient_driver, tdobj, channel_data, atmlst=None,
        with_direct=True):
    """LDA/GGA/meta-GGA ``vref0/vref1`` M matrix and fixed-grid skeleton derivative."""
    mf = tdobj._scf
    xctype = mf._numint._xc_type(mf.xc)
    add_matrix, eval_matrix = _xc_matrix_builders(xctype)
    if xctype == "LDA":
        nvar = 1
    elif xctype == "GGA":
        nvar, pair_potential, pair_cross = 4, gga_pair_potential, gga_pair_kernel_cross
    else:
        nvar, pair_potential, pair_cross = 5, mgga_pair_potential, mgga_pair_kernel_cross
    mol = mf.mol
    ni = mf._numint
    if atmlst is None:
        atmlst = range(mol.natm)
    atmlst = tuple(atmlst)
    _spaces, _amplitudes, densities, blocks, terms = channel_data
    pair_labels = tuple(
        label for label in densities
        if xctype != "LDA" and any(
            term.vref1 and label in (term.target, term.source)
            for term in terms
        )
    )
    pair_density_stack = np.asarray([
        densities[label] for label in pair_labels
    ])
    nao = mol.nao_nr()
    potentials = {label: np.zeros((nao, nao)) for label in densities}
    reference_alpha = np.zeros((nao, nao))
    reference_beta = np.zeros_like(reference_alpha)
    direct = np.zeros((len(atmlst), 3))
    mo = np.asarray(mf.mo_coeff)
    density_alpha, density_beta = _reference_spin_densities(tdobj)
    density_labels, density_stack = _response_density_stack(
        densities, density_alpha, density_beta,
    )
    offsets = mol.offset_nr_by_atom()
    sparse = sparse_context(mf) if xctype != "LDA" else None
    ao_deriv = 1 if xctype == "LDA" else 2

    for ao, mask, weights, _coords in ni.block_loop(
            mol, mf.grids, nao, ao_deriv, max_memory=gradient_driver.max_memory):
        rho0 = ni.eval_rho2(
            mol, ao[0] if xctype == "LDA" else ao, mo, mf.mo_occ,
            mask, xctype, with_lapl=False,
        ) * 0.5
        fref, kref_alpha, kref_beta = _reference_fref_kref(mf, rho0, xctype)
        rho = {
            label: _xc_density(ni, mol, ao, density, mask, xctype, hermi=0)
            for label, density in densities.items()
        }
        pairs, pair_potentials = {}, {}
        if pair_labels:
            pair_values, contracted_pair_ao = pair_feature_batches(ao, pair_density_stack)
            pairs = dict(zip(pair_labels, pair_values))
            pair_potentials = {
                label: pair_potential(fref, pairs[label]) for label in pair_labels
            }
        ordinary_weights = {
            label: np.zeros((nvar, weights.size)) for label in densities
        }
        special_weights = {
            label: np.zeros((4, 4, weights.size)) for label in pair_labels
        }
        reference_weights_alpha = np.zeros((nvar, weights.size))
        reference_weights_beta = np.zeros_like(reference_weights_alpha)

        for term in terms:
            # In LDA the two kernels coincide; no pair-feature correction remains.
            ordinary_coefficient = term.vref0 + term.vref1 if xctype == "LDA" else term.vref0
            if ordinary_coefficient:
                ordinary_weights[term.target] += ordinary_coefficient * lib.einsum(
                    "xyg,yg->xg", fref, rho[term.source],
                )
                ordinary_weights[term.source] += ordinary_coefficient * lib.einsum(
                    "xyg,xg->yg", fref, rho[term.target],
                )
                pair = ordinary_coefficient * lib.einsum(
                    "xg,yg->xyg", rho[term.target], rho[term.source],
                )
                reference_weights_alpha += lib.einsum(
                    "xyg,xyzg->zg", pair, kref_alpha,
                )
                reference_weights_beta += lib.einsum(
                    "xyg,xyzg->zg", pair, kref_beta,
                )
            if xctype != "LDA" and term.vref1:
                special_weights[term.target] += (
                    term.vref1 * pair_potentials[term.source]
                )
                special_weights[term.source] += (
                    term.vref1 * pair_potentials[term.target]
                )
                pair = term.vref1 * pair_cross(
                    pairs[term.target], pairs[term.source],
                )
                reference_weights_alpha += lib.einsum(
                    "xyg,xyzg->zg", pair, kref_alpha,
                )
                reference_weights_beta += lib.einsum(
                    "xyg,xyzg->zg", pair, kref_beta,
                )
        ordinary_weight_stack = np.asarray([
            ordinary_weights[label] for label in density_labels
        ])
        special_weight_stack = np.asarray([
            special_weights[label] for label in pair_labels
        ])

        for label in potentials:
            add_matrix(
                mol, potentials[label], ao,
                ordinary_weights[label] * weights, mask, sparse,
            )
        for label in pair_labels:
            potentials[label] += pair_matrix(
                mol, ao, mask, special_weights[label] * weights, sparse,
            )
        reference_alpha += eval_matrix(
            mol, ao, reference_weights_alpha * weights, mask,
        )
        reference_beta += eval_matrix(
            mol, ao, reference_weights_beta * weights, mask,
        )

        if not with_direct:
            continue
        for k, atom in enumerate(atmlst):
            p0, p1 = offsets[atom][2:]
            for xyz in range(3):
                drho, drho_alpha, drho_beta, ao_delta = (
                    _response_density_derivatives(
                        ao, density_stack, density_labels,
                        p0, p1, xyz, xctype,
                    )
                )
                drho_stack = np.asarray([
                    drho[label] for label in density_labels
                ])
                value = lib.einsum(
                    "nfg,nfg,g->",
                    ordinary_weight_stack, drho_stack, weights,
                )
                value += lib.einsum(
                    "fg,fg,g->",
                    reference_weights_alpha, drho_alpha, weights,
                )
                value += lib.einsum(
                    "fg,fg,g->",
                    reference_weights_beta, drho_beta, weights,
                )
                if pair_labels:
                    value += contract_pair_feature_derivatives(
                        ao, pair_density_stack, ao_delta,
                        contracted_pair_ao, p0, p1,
                        special_weight_stack, weights,
                    )
                direct[k, xyz] += value

    q_alpha, q_beta = _project_channel_potentials(
        tdobj, potentials, blocks,
    )
    _add_reference_q(
        tdobj, q_alpha, q_beta, reference_alpha, reference_beta,
    )
    return XCGradientTerms(q_alpha, q_beta, direct)


def _lda_eval_matrix(mol, ao, weights, mask):
    """LDA potential with the same feature axis as GGA/meta-GGA."""
    return _lda_matrix(ao[0], weights[0])


def _add_lda_matrix(mol, output, ao, weights, mask, sparse):
    output += _lda_eval_matrix(mol, ao, weights, mask)


def _xc_matrix_builders(xctype):
    """Feature-potential builders; LDA retains one density feature."""
    if xctype == "LDA":
        return _add_lda_matrix, _lda_eval_matrix
    if xctype == "GGA":
        return add_gga_matrix, gga_eval_matrix
    if xctype == "MGGA":
        return add_mgga_matrix, mgga_eval_matrix
    raise NotImplementedError("Unsupported XC type %s" % xctype)


def fockz_terms(
        gradient_driver, tdobj, spaces, pz, atmlst=None,
        with_direct=True):
    """LDA/GGA/meta-GGA derivative of ``Pz:Fz`` excluding its explicit Pz projection."""
    mf = tdobj._scf
    xctype = mf._numint._xc_type(mf.xc)
    add_matrix, eval_matrix = _xc_matrix_builders(xctype)
    mol = mf.mol
    ni = mf._numint
    if atmlst is None:
        atmlst = range(mol.natm)
    atmlst = tuple(atmlst)
    density_open = spaces.c_open @ spaces.c_open.T
    pz = 0.5 * (np.asarray(pz) + np.asarray(pz).T)
    nao = mol.nao_nr()
    open_potential = np.zeros((nao, nao))
    reference_alpha = np.zeros((nao, nao))
    reference_beta = np.zeros_like(reference_alpha)
    direct = np.zeros((len(atmlst), 3))
    mo = np.asarray(mf.mo_coeff)
    density_alpha, density_beta = _reference_spin_densities(tdobj)
    density_stack = np.asarray((
        pz, density_open, density_alpha, density_beta,
    ))
    offsets = mol.offset_nr_by_atom()
    sparse = sparse_context(mf) if xctype != "LDA" else None
    ao_deriv = 1 if xctype == "LDA" else 2

    for ao, mask, weights, _coords in ni.block_loop(
            mol, mf.grids, nao, ao_deriv, max_memory=gradient_driver.max_memory):
        rho0 = ni.eval_rho2(
            mol, ao[0] if xctype == "LDA" else ao, mo, mf.mo_occ,
            mask, xctype, with_lapl=False,
        ) * 0.5
        fref, kref_alpha, kref_beta = _reference_fref_kref(mf, rho0, xctype)
        rho_pz = _xc_density(ni, mol, ao, pz, mask, xctype)
        rho_open = _xc_density(ni, mol, ao, density_open, mask, xctype)
        add_matrix(
            mol,
            open_potential,
            ao,
            0.5 * lib.einsum("xyg,yg->xg", fref, rho_pz) * weights,
            mask,
            sparse,
        )
        pair = 0.5 * lib.einsum("xg,yg->xyg", rho_pz, rho_open)
        reference_alpha += eval_matrix(
            mol, ao, lib.einsum("xyg,xyzg->zg", pair, kref_alpha) * weights,
            mask,
        )
        reference_beta += eval_matrix(
            mol, ao, lib.einsum("xyg,xyzg->zg", pair, kref_beta) * weights,
            mask,
        )
        if not with_direct:
            continue
        for k, atom in enumerate(atmlst):
            p0, p1 = offsets[atom][2:]
            derivative_batches = _hermitian_density_derivative_batches(
                ao, density_stack, p0, p1, xctype,
            )
            for xyz, derivatives in enumerate(derivative_batches):
                drho_pz, drho_open, drho_alpha, drho_beta = derivatives
                direct[k, xyz] += 0.5 * lib.einsum(
                    "xg,xyg,yg,g->", drho_pz, fref, rho_open, weights,
                )
                direct[k, xyz] += 0.5 * lib.einsum(
                    "xg,xyg,yg,g->", rho_pz, fref, drho_open, weights,
                )
                direct[k, xyz] += lib.einsum(
                    "xyg,xyzg,zg,g->",
                    pair, kref_alpha, drho_alpha, weights,
                )
                direct[k, xyz] += lib.einsum(
                    "xyg,xyzg,zg,g->",
                    pair, kref_beta, drho_beta, weights,
                )

    q_alpha = np.zeros((mo.shape[1], mo.shape[1]))
    q_beta = np.zeros_like(q_alpha)
    q_alpha[:, spaces.open] += (
        mo.conj().T @ (open_potential + open_potential.T) @ spaces.c_open
    )
    _add_reference_q(
        tdobj, q_alpha, q_beta, reference_alpha, reference_beta,
    )
    return XCGradientTerms(q_alpha, q_beta, direct)


def nobeta_reference_q(tdobj, p0, max_memory=None):
    """Reference-density response of the LDA/GGA/meta-GGA equal-spin common Fock."""
    mf = tdobj._scf
    xctype = mf._numint._xc_type(mf.xc)
    _add_matrix, eval_matrix = _xc_matrix_builders(xctype)
    mo = np.asarray(mf.mo_coeff)
    q_alpha = np.zeros((mo.shape[1], mo.shape[1]))
    q_beta = np.zeros_like(q_alpha)
    if not tdobj.nobeta or getattr(mf, "is_average_occupation_reference", False):
        return q_alpha, q_beta
    if max_memory is None:
        max_memory = tdobj.max_memory
    ni = mf._numint
    mol = mf.mol
    density_alpha, density_beta = _reference_spin_densities(tdobj)
    density0 = 0.5 * (density_alpha + density_beta)
    p0 = 0.5 * (np.asarray(p0) + np.asarray(p0).T)
    matrix_alpha = np.zeros((mol.nao_nr(), mol.nao_nr()))
    matrix_beta = np.zeros_like(matrix_alpha)
    ao_deriv = 1 if xctype == "LDA" else 2
    for ao, mask, weights, _coords in ni.block_loop(
            mol, mf.grids, mol.nao_nr(), ao_deriv, max_memory=max_memory):
        rho_p, rho_alpha, rho_beta, rho_equal = (
            _xc_density(ni, mol, ao, density, mask, xctype)
            for density in (p0, density_alpha, density_beta, density0)
        )
        fxc_actual = ni.eval_xc_eff(
            mf.xc, (rho_alpha, rho_beta), deriv=2,
            xctype=xctype, spin=1,
        )[2]
        fxc_equal = ni.eval_xc_eff(
            mf.xc, (rho_equal, rho_equal), deriv=2,
            xctype=xctype, spin=1,
        )[2]
        equal = 0.25 * (
            fxc_equal[0, :, 0] + fxc_equal[0, :, 1]
            + fxc_equal[1, :, 0] + fxc_equal[1, :, 1]
        )
        actual_alpha = 0.5 * (
            fxc_actual[0, :, 0] + fxc_actual[1, :, 0]
        )
        actual_beta = 0.5 * (
            fxc_actual[0, :, 1] + fxc_actual[1, :, 1]
        )
        matrix_alpha += eval_matrix(
            mol,
            ao,
            lib.einsum("xg,xzg->zg", rho_p, equal - actual_alpha) * weights,
            mask,
        )
        matrix_beta += eval_matrix(
            mol,
            ao,
            lib.einsum("xg,xzg->zg", rho_p, equal - actual_beta) * weights,
            mask,
        )
    _add_reference_q(tdobj, q_alpha, q_beta, matrix_alpha, matrix_beta)
    return q_alpha, q_beta
