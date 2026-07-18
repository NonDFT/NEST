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


# LDA quadrature

def _lda_fref_kref(mf, ao0, mask):
    ni = mf._numint
    rho0 = ni.eval_rho2(
        mf.mol, ao0, mf.mo_coeff, mf.mo_occ, mask, "LDA",
        with_lapl=False,
    ) * 0.5
    fxc, kxc = ni.eval_xc_eff(
        mf.xc, (rho0, rho0), deriv=3, xctype="LDA", spin=1,
    )[2:4]
    fref = 0.5 * (
        fxc[0, 0, 0, 0] - fxc[0, 0, 1, 0]
        - fxc[1, 0, 0, 0] + fxc[1, 0, 1, 0]
    )
    kref_alpha = 0.5 * (
        kxc[0, 0, 0, 0, 0, 0] - kxc[0, 0, 1, 0, 0, 0]
        - kxc[1, 0, 0, 0, 0, 0] + kxc[1, 0, 1, 0, 0, 0]
    )
    kref_beta = 0.5 * (
        kxc[0, 0, 0, 0, 1, 0] - kxc[0, 0, 1, 0, 1, 0]
        - kxc[1, 0, 0, 0, 1, 0] + kxc[1, 0, 1, 0, 1, 0]
    )
    return fref, kref_alpha, kref_beta


def _lda_matrix(ao0, weights):
    return ao0.T @ (ao0 * np.asarray(weights)[:, None])


def _unpack_channel_data(channel_data):
    """Return channel data plus the spin carried by each target orbital.

    Historical ``deltaS=0/-1`` callers use beta-spin targets and pass the
    original five-item tuple.  The spin-raising channel appends ``"alpha"``
    because its sole C->V excitation removes beta spin and creates alpha spin.
    """
    if len(channel_data) == 5:
        return (*channel_data, "beta")
    if len(channel_data) == 6:
        return tuple(channel_data)
    raise ValueError("NTTDA XC channel data must contain five or six items")


def _project_channel_potentials(
        tdobj, potentials, blocks, target_spin="beta"):
    """Project transition-factor potentials for any NTTDA spin channel."""
    mo = np.asarray(tdobj._scf.mo_coeff)
    q_alpha = np.zeros((mo.shape[1], mo.shape[1]))
    q_beta = np.zeros_like(q_alpha)
    for label, (target, source, coefficient) in blocks.items():
        potential = mo.conj().T @ potentials[label] @ mo
        if target_spin == "beta":
            q_beta[:, target] += potential[:, source] @ coefficient.T
            q_alpha[:, source] += potential[target, :].T @ coefficient
        elif target_spin == "alpha":
            q_alpha[:, target] += potential[:, source] @ coefficient.T
            q_beta[:, source] += potential[target, :].T @ coefficient
        else:
            raise ValueError("target_spin must be 'alpha' or 'beta'")
    return q_alpha, q_beta


def _add_reference_q(tdobj, q_alpha, q_beta, matrix_alpha, matrix_beta):
    mf = tdobj._scf
    mo = np.asarray(mf.mo_coeff)
    occupied_alpha = np.flatnonzero(mf.mo_occ > 0)
    occupied_beta = np.flatnonzero(mf.mo_occ == 2)
    q_alpha[:, occupied_alpha] += (
        mo.conj().T @ (matrix_alpha + matrix_alpha.T)
        @ mo[:, occupied_alpha]
    )
    q_beta[:, occupied_beta] += (
        mo.conj().T @ (matrix_beta + matrix_beta.T)
        @ mo[:, occupied_beta]
    )


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


def _xc_density(ni, mol, ao, density, mask, xctype):
    ao_values = ao[0] if xctype == "LDA" else ao
    rho = ni.eval_rho(
        mol, ao_values, density, mask, xctype, hermi=1,
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
    if xctype == "LDA":
        derivatives = derivatives[:, 0]
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


def lda_response_terms(
        gradient_driver, tdobj, channel_data, atmlst=None,
        with_direct=True):
    """Analytic LDA M/direct terms for the ``vref0/vref1`` scalar."""
    mf = tdobj._scf
    mol = mf.mol
    ni = mf._numint
    if atmlst is None:
        atmlst = range(mol.natm)
    atmlst = tuple(atmlst)
    (
        _spaces, _amplitudes, densities, blocks, response_terms, target_spin,
    ) = _unpack_channel_data(channel_data)
    coefficients = tuple(
        (term.target, term.source, term.vref0 + term.vref1)
        for term in response_terms
        if term.vref0 + term.vref1
    )
    nao = mol.nao_nr()
    potentials = {label: np.zeros((nao, nao)) for label in densities}
    reference_alpha = np.zeros((nao, nao))
    reference_beta = np.zeros_like(reference_alpha)
    direct = np.zeros((len(atmlst), 3))
    mo = np.asarray(mf.mo_coeff)
    density_alpha = mo[:, mf.mo_occ > 0] @ mo[:, mf.mo_occ > 0].T
    density_beta = mo[:, mf.mo_occ == 2] @ mo[:, mf.mo_occ == 2].T
    density_labels, density_stack = _response_density_stack(
        densities, density_alpha, density_beta,
    )
    offsets = mol.offset_nr_by_atom()

    for ao, mask, weights, _coords in ni.block_loop(
            mol, mf.grids, nao, 1, max_memory=gradient_driver.max_memory):
        ao0 = ao[0]
        fref, kref_alpha, kref_beta = _lda_fref_kref(mf, ao0, mask)
        rho = {
            label: ni.eval_rho(
                mol, ao0, density, mask, "LDA", hermi=0,
                with_lapl=False,
            )
            for label, density in densities.items()
        }
        potential_weights = {
            label: np.zeros_like(weights) for label in densities
        }
        pair_alpha = np.zeros_like(weights)
        pair_beta = np.zeros_like(weights)
        for target, source, coefficient in coefficients:
            potential_weights[target] += coefficient * fref * rho[source]
            potential_weights[source] += coefficient * fref * rho[target]
            pair = coefficient * rho[target] * rho[source]
            pair_alpha += kref_alpha * pair
            pair_beta += kref_beta * pair
        potential_weight_stack = np.asarray([
            potential_weights[label] for label in density_labels
        ])
        for label in potentials:
            potentials[label] += _lda_matrix(
                ao0, weights * potential_weights[label],
            )
        reference_alpha += _lda_matrix(ao0, weights * pair_alpha)
        reference_beta += _lda_matrix(ao0, weights * pair_beta)

        if not with_direct:
            continue
        for k, atom in enumerate(atmlst):
            p0, p1 = offsets[atom][2:]
            for xyz in range(3):
                drho, drho_alpha, drho_beta, _ao_delta = (
                    _response_density_derivatives(
                        ao, density_stack, density_labels,
                        p0, p1, xyz, "LDA",
                    )
                )
                drho_stack = np.asarray([
                    drho[label] for label in density_labels
                ])
                value = lib.einsum(
                    "ng,ng,g->",
                    potential_weight_stack, drho_stack, weights,
                )
                value += lib.einsum(
                    "g,g,g->", pair_alpha, drho_alpha, weights,
                )
                value += lib.einsum(
                    "g,g,g->", pair_beta, drho_beta, weights,
                )
                direct[k, xyz] += value

    q_alpha, q_beta = _project_channel_potentials(
        tdobj, potentials, blocks, target_spin=target_spin,
    )
    _add_reference_q(
        tdobj, q_alpha, q_beta, reference_alpha, reference_beta,
    )
    return XCGradientTerms(q_alpha, q_beta, direct)


def lda_fockz_terms(
        gradient_driver, tdobj, spaces, pz, atmlst=None,
        with_direct=True):
    """LDA response/direct derivative of ``Pz:Fz`` excluding Pz projection."""
    mf = tdobj._scf
    mol = mf.mol
    ni = mf._numint
    if atmlst is None:
        atmlst = range(mol.natm)
    atmlst = tuple(atmlst)
    density_open = spaces.c_open @ spaces.c_open.T
    pz = np.asarray(pz)
    pz_symmetric = 0.5 * (pz + pz.T)
    nao = mol.nao_nr()
    open_potential = np.zeros((nao, nao))
    reference_alpha = np.zeros((nao, nao))
    reference_beta = np.zeros_like(reference_alpha)
    direct = np.zeros((len(atmlst), 3))
    mo = np.asarray(mf.mo_coeff)
    density_alpha = mo[:, mf.mo_occ > 0] @ mo[:, mf.mo_occ > 0].T
    density_beta = mo[:, mf.mo_occ == 2] @ mo[:, mf.mo_occ == 2].T
    density_stack = np.asarray((
        pz_symmetric, density_open, density_alpha, density_beta,
    ))
    offsets = mol.offset_nr_by_atom()

    for ao, mask, weights, _coords in ni.block_loop(
            mol, mf.grids, nao, 1, max_memory=gradient_driver.max_memory):
        ao0 = ao[0]
        fref, kref_alpha, kref_beta = _lda_fref_kref(mf, ao0, mask)
        rho_pz = ni.eval_rho(
            mol, ao0, pz_symmetric, mask, "LDA", hermi=1,
            with_lapl=False,
        )
        rho_open = ni.eval_rho(
            mol, ao0, density_open, mask, "LDA", hermi=1,
            with_lapl=False,
        )
        open_potential += _lda_matrix(
            ao0, 0.5 * weights * fref * rho_pz,
        )
        pair = 0.5 * rho_pz * rho_open
        reference_alpha += _lda_matrix(
            ao0, weights * kref_alpha * pair,
        )
        reference_beta += _lda_matrix(
            ao0, weights * kref_beta * pair,
        )
        if not with_direct:
            continue
        for k, atom in enumerate(atmlst):
            p0, p1 = offsets[atom][2:]
            derivative_batches = _hermitian_density_derivative_batches(
                ao, density_stack, p0, p1, "LDA",
            )
            for xyz, derivatives in enumerate(derivative_batches):
                drho_pz, drho_open, drho_alpha, drho_beta = (
                    derivatives[:, 0]
                )
                direct[k, xyz] += 0.5 * np.dot(
                    weights,
                    fref * (
                        drho_pz * rho_open + rho_pz * drho_open
                    )
                    + rho_pz * rho_open * (
                        kref_alpha * drho_alpha
                        + kref_beta * drho_beta
                    ),
                )

    q_alpha = np.zeros((mo.shape[1], mo.shape[1]))
    q_beta = np.zeros_like(q_alpha)
    q_alpha[:, spaces.open] += (
        mo.conj().T @ (open_potential + open_potential.T)
        @ spaces.c_open
    )
    _add_reference_q(
        tdobj, q_alpha, q_beta, reference_alpha, reference_beta,
    )
    return XCGradientTerms(q_alpha, q_beta, direct)


def lda_nobeta_reference_q(tdobj, p0, max_memory=None):
    """Reference-density correction for the equal-spin ``nobeta`` Fock."""
    mf = tdobj._scf
    mo = np.asarray(mf.mo_coeff)
    nmo = mo.shape[1]
    q_alpha = np.zeros((nmo, nmo))
    q_beta = np.zeros_like(q_alpha)
    if not tdobj.nobeta:
        return q_alpha, q_beta
    if max_memory is None:
        max_memory = tdobj.max_memory
    ni = mf._numint
    mol = mf.mol
    nao = mol.nao_nr()
    density_alpha = mo[:, mf.mo_occ > 0] @ mo[:, mf.mo_occ > 0].T
    density_beta = mo[:, mf.mo_occ == 2] @ mo[:, mf.mo_occ == 2].T
    density0 = 0.5 * (density_alpha + density_beta)
    p0 = 0.5 * (np.asarray(p0) + np.asarray(p0).T)
    matrix_alpha = np.zeros((nao, nao))
    matrix_beta = np.zeros_like(matrix_alpha)
    for ao, mask, weights, _coords in ni.block_loop(
            mol, mf.grids, nao, 1, max_memory=max_memory):
        ao0 = ao[0]
        rho_p = ni.eval_rho(
            mol, ao0, p0, mask, "LDA", hermi=1, with_lapl=False,
        )
        rho_alpha = ni.eval_rho(
            mol, ao0, density_alpha, mask, "LDA", hermi=1,
            with_lapl=False,
        )
        rho_beta = ni.eval_rho(
            mol, ao0, density_beta, mask, "LDA", hermi=1,
            with_lapl=False,
        )
        rho0 = ni.eval_rho(
            mol, ao0, density0, mask, "LDA", hermi=1,
            with_lapl=False,
        )
        fxc_actual = ni.eval_xc_eff(
            mf.xc, (rho_alpha, rho_beta), deriv=2,
            xctype="LDA", spin=1,
        )[2]
        fxc_equal = ni.eval_xc_eff(
            mf.xc, (rho0, rho0), deriv=2,
            xctype="LDA", spin=1,
        )[2]
        equal_derivative = 0.25 * (
            fxc_equal[0, 0, 0, 0] + fxc_equal[0, 0, 1, 0]
            + fxc_equal[1, 0, 0, 0] + fxc_equal[1, 0, 1, 0]
        )
        actual_alpha = 0.5 * (
            fxc_actual[0, 0, 0, 0] + fxc_actual[1, 0, 0, 0]
        )
        actual_beta = 0.5 * (
            fxc_actual[0, 0, 1, 0] + fxc_actual[1, 0, 1, 0]
        )
        matrix_alpha += _lda_matrix(
            ao0, weights * rho_p * (equal_derivative - actual_alpha),
        )
        matrix_beta += _lda_matrix(
            ao0, weights * rho_p * (equal_derivative - actual_beta),
        )
    _add_reference_q(
        tdobj, q_alpha, q_beta, matrix_alpha, matrix_beta,
    )
    return q_alpha, q_beta


# GGA quadrature

def _gga_fref_kref(mf, rho0):
    fxc, kxc = mf._numint.eval_xc_eff(
        mf.xc, (rho0, rho0), deriv=3, xctype="GGA", spin=1,
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


def gga_response_terms(
        gradient_driver, tdobj, channel_data, atmlst=None,
        with_direct=True):
    """GGA ``vref0/vref1`` M matrix and fixed-grid skeleton derivative."""
    mf = tdobj._scf
    mol = mf.mol
    ni = mf._numint
    if atmlst is None:
        atmlst = range(mol.natm)
    atmlst = tuple(atmlst)
    (
        _spaces, _amplitudes, densities, blocks, terms, target_spin,
    ) = _unpack_channel_data(channel_data)
    pair_labels = tuple(
        label for label in densities
        if any(
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
    density_alpha = mo[:, mf.mo_occ > 0] @ mo[:, mf.mo_occ > 0].T
    density_beta = mo[:, mf.mo_occ == 2] @ mo[:, mf.mo_occ == 2].T
    density_labels, density_stack = _response_density_stack(
        densities, density_alpha, density_beta,
    )
    offsets = mol.offset_nr_by_atom()
    sparse = sparse_context(mf)

    for ao, mask, weights, _coords in ni.block_loop(
            mol, mf.grids, nao, 2, max_memory=gradient_driver.max_memory):
        rho0 = ni.eval_rho2(
            mol, ao, mo, mf.mo_occ, mask, "GGA", with_lapl=False,
        ) * 0.5
        fref, kref_alpha, kref_beta = _gga_fref_kref(mf, rho0)
        rho = {
            label: ni.eval_rho(
                mol, ao, density, mask, "GGA", hermi=0,
                with_lapl=False,
            )
            for label, density in densities.items()
        }
        if pair_labels:
            pair_values, contracted_pair_ao = pair_feature_batches(
                ao, pair_density_stack,
            )
        else:
            pair_values = ()
            contracted_pair_ao = None
        pairs = dict(zip(pair_labels, pair_values))
        pair_potentials = {
            label: gga_pair_potential(fref, pairs[label])
            for label in pair_labels
        }
        ordinary_weights = {
            label: np.zeros((4, weights.size)) for label in densities
        }
        special_weights = {
            label: np.zeros((4, 4, weights.size)) for label in pair_labels
        }
        reference_weights_alpha = np.zeros((4, weights.size))
        reference_weights_beta = np.zeros_like(reference_weights_alpha)

        for term in terms:
            if term.vref0:
                ordinary_weights[term.target] += term.vref0 * lib.einsum(
                    "xyg,yg->xg", fref, rho[term.source],
                )
                ordinary_weights[term.source] += term.vref0 * lib.einsum(
                    "xyg,xg->yg", fref, rho[term.target],
                )
                pair = term.vref0 * lib.einsum(
                    "xg,yg->xyg", rho[term.target], rho[term.source],
                )
                reference_weights_alpha += lib.einsum(
                    "xyg,xyzg->zg", pair, kref_alpha,
                )
                reference_weights_beta += lib.einsum(
                    "xyg,xyzg->zg", pair, kref_beta,
                )
            if term.vref1:
                special_weights[term.target] += (
                    term.vref1 * pair_potentials[term.source]
                )
                special_weights[term.source] += (
                    term.vref1 * pair_potentials[term.target]
                )
                pair = term.vref1 * gga_pair_kernel_cross(
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
            add_gga_matrix(
                mol, potentials[label], ao,
                ordinary_weights[label] * weights, mask, sparse,
            )
        for label in pair_labels:
            potentials[label] += pair_matrix(
                mol, ao, mask, special_weights[label] * weights, sparse,
            )
        reference_alpha += gga_eval_matrix(
            mol, ao, reference_weights_alpha * weights, mask,
        )
        reference_beta += gga_eval_matrix(
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
                        p0, p1, xyz, "GGA",
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
        tdobj, potentials, blocks, target_spin=target_spin,
    )
    _add_reference_q(
        tdobj, q_alpha, q_beta, reference_alpha, reference_beta,
    )
    return XCGradientTerms(q_alpha, q_beta, direct)


def gga_fockz_terms(
        gradient_driver, tdobj, spaces, pz, atmlst=None,
        with_direct=True):
    """GGA derivative of ``Pz:Fz`` excluding its explicit Pz projection."""
    mf = tdobj._scf
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
    density_alpha = mo[:, mf.mo_occ > 0] @ mo[:, mf.mo_occ > 0].T
    density_beta = mo[:, mf.mo_occ == 2] @ mo[:, mf.mo_occ == 2].T
    density_stack = np.asarray((
        pz, density_open, density_alpha, density_beta,
    ))
    offsets = mol.offset_nr_by_atom()
    sparse = sparse_context(mf)

    for ao, mask, weights, _coords in ni.block_loop(
            mol, mf.grids, nao, 2, max_memory=gradient_driver.max_memory):
        rho0 = ni.eval_rho2(
            mol, ao, mo, mf.mo_occ, mask, "GGA", with_lapl=False,
        ) * 0.5
        fref, kref_alpha, kref_beta = _gga_fref_kref(mf, rho0)
        rho_pz = ni.eval_rho(
            mol, ao, pz, mask, "GGA", hermi=1, with_lapl=False,
        )
        rho_open = ni.eval_rho(
            mol, ao, density_open, mask, "GGA", hermi=1,
            with_lapl=False,
        )
        add_gga_matrix(
            mol,
            open_potential,
            ao,
            0.5 * lib.einsum("xyg,yg->xg", fref, rho_pz) * weights,
            mask,
            sparse,
        )
        pair = 0.5 * lib.einsum("xg,yg->xyg", rho_pz, rho_open)
        reference_alpha += gga_eval_matrix(
            mol, ao, lib.einsum("xyg,xyzg->zg", pair, kref_alpha) * weights,
            mask,
        )
        reference_beta += gga_eval_matrix(
            mol, ao, lib.einsum("xyg,xyzg->zg", pair, kref_beta) * weights,
            mask,
        )
        if not with_direct:
            continue
        for k, atom in enumerate(atmlst):
            p0, p1 = offsets[atom][2:]
            derivative_batches = _hermitian_density_derivative_batches(
                ao, density_stack, p0, p1, "GGA",
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


def gga_nobeta_reference_q(tdobj, p0, max_memory=None):
    """Reference-density response of the GGA equal-spin common Fock."""
    mf = tdobj._scf
    mo = np.asarray(mf.mo_coeff)
    q_alpha = np.zeros((mo.shape[1], mo.shape[1]))
    q_beta = np.zeros_like(q_alpha)
    if not tdobj.nobeta:
        return q_alpha, q_beta
    if max_memory is None:
        max_memory = tdobj.max_memory
    ni = mf._numint
    mol = mf.mol
    density_alpha = mo[:, mf.mo_occ > 0] @ mo[:, mf.mo_occ > 0].T
    density_beta = mo[:, mf.mo_occ == 2] @ mo[:, mf.mo_occ == 2].T
    density0 = 0.5 * (density_alpha + density_beta)
    p0 = 0.5 * (np.asarray(p0) + np.asarray(p0).T)
    matrix_alpha = np.zeros((mol.nao_nr(), mol.nao_nr()))
    matrix_beta = np.zeros_like(matrix_alpha)
    for ao, mask, weights, _coords in ni.block_loop(
            mol, mf.grids, mol.nao_nr(), 2, max_memory=max_memory):
        rho_p = ni.eval_rho(
            mol, ao, p0, mask, "GGA", hermi=1, with_lapl=False,
        )
        rho_alpha = ni.eval_rho(
            mol, ao, density_alpha, mask, "GGA", hermi=1,
            with_lapl=False,
        )
        rho_beta = ni.eval_rho(
            mol, ao, density_beta, mask, "GGA", hermi=1,
            with_lapl=False,
        )
        rho_equal = ni.eval_rho(
            mol, ao, density0, mask, "GGA", hermi=1, with_lapl=False,
        )
        fxc_actual = ni.eval_xc_eff(
            mf.xc, (rho_alpha, rho_beta), deriv=2,
            xctype="GGA", spin=1,
        )[2]
        fxc_equal = ni.eval_xc_eff(
            mf.xc, (rho_equal, rho_equal), deriv=2,
            xctype="GGA", spin=1,
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
        matrix_alpha += gga_eval_matrix(
            mol,
            ao,
            lib.einsum("xg,xzg->zg", rho_p, equal - actual_alpha) * weights,
            mask,
        )
        matrix_beta += gga_eval_matrix(
            mol,
            ao,
            lib.einsum("xg,xzg->zg", rho_p, equal - actual_beta) * weights,
            mask,
        )
    _add_reference_q(tdobj, q_alpha, q_beta, matrix_alpha, matrix_beta)
    return q_alpha, q_beta


# meta-GGA quadrature

def _mgga_fref_kref(mf, rho0):
    fxc, kxc = mf._numint.eval_xc_eff(
        mf.xc, (rho0, rho0), deriv=3, xctype="MGGA", spin=1,
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


def mgga_response_terms(
        gradient_driver, tdobj, channel_data, atmlst=None,
        with_direct=True):
    """MGGA ``vref0/vref1`` M matrix and fixed-grid skeleton derivative."""
    mf = tdobj._scf
    mol = mf.mol
    ni = mf._numint
    if atmlst is None:
        atmlst = range(mol.natm)
    atmlst = tuple(atmlst)
    (
        _spaces, _amplitudes, densities, blocks, terms, target_spin,
    ) = _unpack_channel_data(channel_data)
    pair_labels = tuple(
        label for label in densities
        if any(
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
    density_alpha = mo[:, mf.mo_occ > 0] @ mo[:, mf.mo_occ > 0].T
    density_beta = mo[:, mf.mo_occ == 2] @ mo[:, mf.mo_occ == 2].T
    density_labels, density_stack = _response_density_stack(
        densities, density_alpha, density_beta,
    )
    offsets = mol.offset_nr_by_atom()
    sparse = sparse_context(mf)

    for ao, mask, weights, _coords in ni.block_loop(
            mol, mf.grids, nao, 2, max_memory=gradient_driver.max_memory):
        rho0 = ni.eval_rho2(
            mol, ao, mo, mf.mo_occ, mask, "MGGA", with_lapl=False,
        ) * 0.5
        fref, kref_alpha, kref_beta = _mgga_fref_kref(mf, rho0)
        rho = {
            label: ni.eval_rho(
                mol, ao, density, mask, "MGGA", hermi=0,
                with_lapl=False,
            )
            for label, density in densities.items()
        }
        if pair_labels:
            pair_values, contracted_pair_ao = pair_feature_batches(
                ao, pair_density_stack,
            )
        else:
            pair_values = ()
            contracted_pair_ao = None
        pairs = dict(zip(pair_labels, pair_values))
        pair_potentials = {
            label: mgga_pair_potential(fref, pairs[label])
            for label in pair_labels
        }
        ordinary_weights = {
            label: np.zeros((5, weights.size)) for label in densities
        }
        special_weights = {
            label: np.zeros((4, 4, weights.size)) for label in pair_labels
        }
        reference_weights_alpha = np.zeros((5, weights.size))
        reference_weights_beta = np.zeros_like(reference_weights_alpha)

        for term in terms:
            if term.vref0:
                ordinary_weights[term.target] += term.vref0 * lib.einsum(
                    "xyg,yg->xg", fref, rho[term.source],
                )
                ordinary_weights[term.source] += term.vref0 * lib.einsum(
                    "xyg,xg->yg", fref, rho[term.target],
                )
                pair = term.vref0 * lib.einsum(
                    "xg,yg->xyg", rho[term.target], rho[term.source],
                )
                reference_weights_alpha += lib.einsum(
                    "xyg,xyzg->zg", pair, kref_alpha,
                )
                reference_weights_beta += lib.einsum(
                    "xyg,xyzg->zg", pair, kref_beta,
                )
            if term.vref1:
                special_weights[term.target] += (
                    term.vref1 * pair_potentials[term.source]
                )
                special_weights[term.source] += (
                    term.vref1 * pair_potentials[term.target]
                )
                pair = term.vref1 * mgga_pair_kernel_cross(
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
            add_mgga_matrix(
                mol, potentials[label], ao,
                ordinary_weights[label] * weights, mask, sparse,
            )
        for label in pair_labels:
            potentials[label] += pair_matrix(
                mol, ao, mask, special_weights[label] * weights, sparse,
            )
        reference_alpha += mgga_eval_matrix(
            mol, ao, reference_weights_alpha * weights, mask,
        )
        reference_beta += mgga_eval_matrix(
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
                        p0, p1, xyz, "MGGA",
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
        tdobj, potentials, blocks, target_spin=target_spin,
    )
    _add_reference_q(
        tdobj, q_alpha, q_beta, reference_alpha, reference_beta,
    )
    return XCGradientTerms(q_alpha, q_beta, direct)


def mgga_fockz_terms(
        gradient_driver, tdobj, spaces, pz, atmlst=None,
        with_direct=True):
    """MGGA derivative of ``Pz:Fz`` excluding its explicit Pz projection."""
    mf = tdobj._scf
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
    density_alpha = mo[:, mf.mo_occ > 0] @ mo[:, mf.mo_occ > 0].T
    density_beta = mo[:, mf.mo_occ == 2] @ mo[:, mf.mo_occ == 2].T
    density_stack = np.asarray((
        pz, density_open, density_alpha, density_beta,
    ))
    offsets = mol.offset_nr_by_atom()
    sparse = sparse_context(mf)

    for ao, mask, weights, _coords in ni.block_loop(
            mol, mf.grids, nao, 2, max_memory=gradient_driver.max_memory):
        rho0 = ni.eval_rho2(
            mol, ao, mo, mf.mo_occ, mask, "MGGA", with_lapl=False,
        ) * 0.5
        fref, kref_alpha, kref_beta = _mgga_fref_kref(mf, rho0)
        rho_pz = ni.eval_rho(
            mol, ao, pz, mask, "MGGA", hermi=1, with_lapl=False,
        )
        rho_open = ni.eval_rho(
            mol, ao, density_open, mask, "MGGA", hermi=1,
            with_lapl=False,
        )
        add_mgga_matrix(
            mol,
            open_potential,
            ao,
            0.5 * lib.einsum("xyg,yg->xg", fref, rho_pz) * weights,
            mask,
            sparse,
        )
        pair = 0.5 * lib.einsum("xg,yg->xyg", rho_pz, rho_open)
        reference_alpha += mgga_eval_matrix(
            mol, ao, lib.einsum("xyg,xyzg->zg", pair, kref_alpha) * weights,
            mask,
        )
        reference_beta += mgga_eval_matrix(
            mol, ao, lib.einsum("xyg,xyzg->zg", pair, kref_beta) * weights,
            mask,
        )
        if not with_direct:
            continue
        for k, atom in enumerate(atmlst):
            p0, p1 = offsets[atom][2:]
            derivative_batches = _hermitian_density_derivative_batches(
                ao, density_stack, p0, p1, "MGGA",
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


def mgga_nobeta_reference_q(tdobj, p0, max_memory=None):
    """Reference-density response of the MGGA equal-spin common Fock."""
    mf = tdobj._scf
    mo = np.asarray(mf.mo_coeff)
    q_alpha = np.zeros((mo.shape[1], mo.shape[1]))
    q_beta = np.zeros_like(q_alpha)
    if not tdobj.nobeta:
        return q_alpha, q_beta
    if max_memory is None:
        max_memory = tdobj.max_memory
    ni = mf._numint
    mol = mf.mol
    density_alpha = mo[:, mf.mo_occ > 0] @ mo[:, mf.mo_occ > 0].T
    density_beta = mo[:, mf.mo_occ == 2] @ mo[:, mf.mo_occ == 2].T
    density0 = 0.5 * (density_alpha + density_beta)
    p0 = 0.5 * (np.asarray(p0) + np.asarray(p0).T)
    matrix_alpha = np.zeros((mol.nao_nr(), mol.nao_nr()))
    matrix_beta = np.zeros_like(matrix_alpha)
    for ao, mask, weights, _coords in ni.block_loop(
            mol, mf.grids, mol.nao_nr(), 2, max_memory=max_memory):
        rho_p = ni.eval_rho(
            mol, ao, p0, mask, "MGGA", hermi=1, with_lapl=False,
        )
        rho_alpha = ni.eval_rho(
            mol, ao, density_alpha, mask, "MGGA", hermi=1,
            with_lapl=False,
        )
        rho_beta = ni.eval_rho(
            mol, ao, density_beta, mask, "MGGA", hermi=1,
            with_lapl=False,
        )
        rho_equal = ni.eval_rho(
            mol, ao, density0, mask, "MGGA", hermi=1, with_lapl=False,
        )
        fxc_actual = ni.eval_xc_eff(
            mf.xc, (rho_alpha, rho_beta), deriv=2,
            xctype="MGGA", spin=1,
        )[2]
        fxc_equal = ni.eval_xc_eff(
            mf.xc, (rho_equal, rho_equal), deriv=2,
            xctype="MGGA", spin=1,
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
        matrix_alpha += mgga_eval_matrix(
            mol,
            ao,
            lib.einsum("xg,xzg->zg", rho_p, equal - actual_alpha) * weights,
            mask,
        )
        matrix_beta += mgga_eval_matrix(
            mol,
            ao,
            lib.einsum("xg,xzg->zg", rho_p, equal - actual_beta) * weights,
            mask,
        )
    _add_reference_q(tdobj, q_alpha, q_beta, matrix_alpha, matrix_beta)
    return q_alpha, q_beta
