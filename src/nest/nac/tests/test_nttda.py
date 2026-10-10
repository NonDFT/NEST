# Copyright 2026 The NEST Developers. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Independent determinant algebra and nuclear finite differences for NTTDA NAC."""

import copy
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
from pyscf import dft, gto

from nest import aocscf  # noqa: F401 -- registers average_occ
from nest.nac import nttda as nac_module
from nest.nac._nttda_overlap import _one_body, _overlaps, _match_roots, overlap_matrix, wavefunction
from nest.nttda import NTTDA
from nest.nttda.nttda import _orbital_indices, _split_amplitudes, transition_dm


@pytest.fixture(scope='module')
def references():
    cache = {}

    def reference(xc='HF', average=False):
        key = xc, average
        if key not in cache:
            mol = gto.M(
                atom='N 0 0 0; O 0 0 1.20; H 0 0.90 -0.20',
                basis='sto-3g',
                spin=2,
                unit='Bohr',
                verbose=0,
            )
            mf = dft.ROKS(mol)
            if average:
                mf = mf.average_occ()
            mf.set(xc=xc, conv_tol=1e-14, conv_tol_grad=1e-11, max_cycle=200, verbose=0)
            mf.grids.level = 0
            mf.kernel()
            assert mf.converged
            cache[key] = mf
        return cache[key]

    return reference


def make_td(mf, delta_s, nobeta=False):
    # Resolve the complete small raising space to avoid marginal Davidson
    # convergence of the low roots at displaced geometries.
    nc, nv = np.count_nonzero(mf.mo_occ == 2), np.count_nonzero(mf.mo_occ == 0)
    td = (
        NTTDA(mf)
        .set(
            deltaS=delta_s,
            nobeta=nobeta,
            nstates=nc * nv if delta_s == 1 else 3,
            conv_tol=1e-9,
            lindep=1e-20,
            max_cycle=200,
            verbose=0,
        )
        .run()
    )
    assert all(td.converged)
    return td


@pytest.mark.parametrize('nopen,delta_s', [(0, 1), (1, 0), (1, 1), (2, -1), (2, 0), (2, 1), (3, -1), (3, 0)])
def test_spin_adaptation_and_interstate_density(nopen, delta_s):
    """Check tensor phases/densities against explicit fermionic operators."""
    nc, nv = 2, 2
    nmo = nc + nopen + nv
    size = nc * nv if delta_s == 1 else nc * nopen + 2 * nc * nv + 1 + nopen * nv
    if delta_s == -1:
        size = (nc + nopen) * (nopen + nv)
    rng = np.random.default_rng(42 + nopen)
    x = rng.normal(size=(size, 2))
    if delta_s == -1:
        for k in range(2):
            oo = x[:, k].reshape(nc + nopen, nopen + nv)[nc:, :nopen]
            oo -= np.eye(nopen) * np.trace(oo) / nopen
    x = np.linalg.qr(x)[0].T
    # Deliberately interleave occupation spaces to catch ordered-index assumptions.
    occ = np.array([2] * nc + [1] * nopen + [0] * nv)[rng.permutation(nmo)]
    td = SimpleNamespace(deltaS=delta_s, _scf=SimpleNamespace(mo_occ=occ), xy=[(v, 0) for v in x])
    states = [wavefunction(td, xy) for xy in td.xy]
    np.testing.assert_allclose(_overlaps(states, states, np.eye(nmo)), np.eye(2), atol=2e-14, rtol=0)
    for state in states:
        raised = {}
        for p in range(nmo):
            for det, value in _one_body(state, p, nmo + p).items():
                raised[det] = raised.get(det, 0) + value
        assert max(map(abs, raised.values()), default=0) < 2e-14
        for det in state:
            assert (det & ((1 << nmo) - 1)).bit_count() == nc + nopen + delta_s
            assert (det >> nmo).bit_count() == nc - delta_s
    gamma = np.zeros((nmo, nmo))
    for p in range(nmo):
        for q in range(nmo):
            for offset in (0, nmo):
                applied = _one_body(states[1], p + offset, q + offset)
                gamma[p, q] += sum(states[0].get(det, 0) * c for det, c in applied.items())
    order = np.concatenate(_orbital_indices(td))
    np.testing.assert_allclose(transition_dm(td, 0, 1)[np.ix_(order, order)], gamma, atol=2e-14, rtol=0)


def test_rectangular_mo_overlap_and_empty_beta_sector():
    np.testing.assert_allclose(_overlaps([{1: 1.0}], [{4: 1.0}], np.array([[0, 0, 0.75], [0, 0, 0]])), [[0.75]])


@pytest.mark.parametrize('delta_s', [-1, 0, 1])
@pytest.mark.parametrize(
    'xc,average,nobeta',
    [
        ('HF', False, False),
        ('HF', True, False),
        ('PBE', False, False),
        ('PBE', True, False),
        ('SVWN', False, False),
        ('TPSS', False, False),
        ('M06-2X', False, True),
        ('CAM-B3LYP', False, False),
    ],
)
def test_full_nac_matches_awf_finite_difference(references, delta_s, xc, average, nobeta):
    td = make_td(references(xc, average), delta_s, nobeta)
    nac = td.NAC()
    analytic = nac.kernel(1, 2, atmlst=[0], ediff=True)
    saved = [td.mol.atom_coords(), td._scf.mo_coeff.copy(), td.e.copy()]
    if td._scf.grids.coords is not None:
        saved.append(td._scf.grids.coords.copy())
    numeric = nac.finite_difference(1, 2, atmlst=[0], step=2e-4)
    error = np.max(np.abs(analytic - numeric))
    print('NAC FD', delta_s, xc, 'AOCSCF' if average else 'ROKS', 'nobeta', nobeta, 'max error', error)
    np.testing.assert_allclose(analytic, numeric, atol=3e-5, rtol=0)
    assert np.linalg.norm(analytic) > 1e-3
    # The overlap check must not mutate the original geometry, orbitals or grids.
    current = [td.mol.atom_coords(), td._scf.mo_coeff, td.e]
    if td._scf.grids.coords is not None:
        current.append(td._scf.grids.coords)
    for before, after in zip(saved, current):
        np.testing.assert_array_equal(before, after)
    assert nac.nttda_nac_details['csf_residual'] < 1e-11


@pytest.mark.parametrize('delta_s', [-1, 0, 1])
def test_pair_exchange_phase_gap_scaling_and_etfs(references, delta_s):
    td = make_td(references(), delta_s)
    nac = td.nac_method()
    assert isinstance(nac, nac_module.NAC)
    full = nac.kernel(1, 2, ediff=True)
    details = nac.nttda_nac_details
    gap = td.e[1] - td.e[0]
    np.testing.assert_allclose(nac.kernel(2, 1), -full, atol=2e-8, rtol=0)
    np.testing.assert_allclose(nac.kernel(1, 2, ediff=False), gap * full, atol=2e-8, rtol=0)
    np.testing.assert_allclose(nac.kernel(2, 1), gap * full, atol=2e-8, rtol=0)
    etf = nac.kernel(1, 2, ediff=True, use_etfs=True)
    np.testing.assert_allclose(full - etf, details['csf_ao'], atol=2e-8, rtol=0)
    np.testing.assert_allclose(etf.sum(axis=0), 0, atol=1e-8, rtol=0)
    np.testing.assert_allclose(etf, details['hf_numerator'] / gap + details['csf_orbital'], atol=2e-8, rtol=0)
    if delta_s == 1:
        np.testing.assert_array_equal(details['csf_orbital'], 0)
    else:
        assert np.linalg.norm(details['csf_orbital']) > 1e-4
    td.xy[1] = (-td.xy[1][0], 0)
    np.testing.assert_allclose(nac.kernel(1, 2, use_etfs=False), -full, atol=2e-8, rtol=0)


@pytest.mark.parametrize('delta_s', [-1, 0, 1])
def test_polarization_and_atom_selection(references, delta_s):
    td = make_td(references(), delta_s)
    nac = td.NAC()
    xi, xj = td.xy[:2]
    numerator = nac_module.get_hf_interstate_numerator(nac, xi, xj, [0])
    np.testing.assert_allclose(numerator, nac_module.get_hf_interstate_numerator(nac, xj, xi, [0]), atol=1e-12)
    diagonal = nac_module.get_hf_interstate_numerator(nac, xi, xi, [0])
    np.testing.assert_allclose(diagonal, td.Gradients().grad_elec(xi, [0]), atol=1e-9)
    full = nac.kernel(1, 2)
    np.testing.assert_allclose(nac.kernel(atmlst=[2, 0]), full[[2, 0]], atol=2e-9)
    assert nac.kernel(atmlst=[]).shape == (0, 3)


def test_input_validation_and_near_degeneracy(references):
    td = make_td(references(), 0)
    nac = td.NAC()
    for pair in ((None, None), (0, 2), (1, 4), (1, 1), (1.5, 2)):
        with pytest.raises(ValueError):
            nac.kernel(*pair)
    with patch.object(td._scf, 'converged', False), pytest.raises(RuntimeError, match='converge'):
        nac.kernel(1, 2)
    with patch.object(td, 'converged', [True, False, True]), pytest.raises(RuntimeError, match='converge'):
        nac.kernel(1, 2)
    with (
        patch.object(td._scf, 'mo_coeff', td._scf.mo_coeff.astype(complex)),
        pytest.raises(NotImplementedError, match='real'),
    ):
        nac.kernel(1, 2)
    for attribute in ('with_df', 'with_x2c', 'with_solvent'):
        with patch.object(td._scf, attribute, object(), create=True), pytest.raises(NotImplementedError):
            nac.kernel(1, 2)
    for step in (0, -1, np.nan):
        with pytest.raises(ValueError, match='step'):
            nac.finite_difference(1, 2, step=step)
    for atoms in ([-1], [td.mol.natm], [0.5]):
        with pytest.raises(ValueError, match='atmlst'):
            nac.kernel(1, 2, atmlst=atoms)
    td.e[1] = td.e[0] + 1e-12
    with pytest.raises(ZeroDivisionError, match='gap'):
        nac.kernel(1, 2, ediff=True)
    assert np.isfinite(nac.kernel(1, 2, atmlst=[0], ediff=False)).all()


@pytest.mark.parametrize('delta_s', [-1, 0, 1])
def test_same_space_orbital_rotations_leave_nac_unchanged(references, delta_s):
    td = make_td(references(), delta_s)
    expected = td.NAC().kernel(1, 2, atmlst=[0], ediff=True)
    rotated = copy.copy(td)
    rotated._scf = td._scf.copy()
    rotated._scf.mo_coeff = td._scf.mo_coeff.copy()
    rng = np.random.default_rng(73)
    rotations = []
    for indices in _orbital_indices(td):
        u = np.linalg.qr(rng.normal(size=(len(indices), len(indices))))[0]
        rotated._scf.mo_coeff[:, indices] = td._scf.mo_coeff[:, indices] @ u
        rotations.append(u)
    c, o, v = rotations
    nc, no, nv = map(len, rotations)
    rotated.xy = []
    for xy in td.xy:
        co, cv, oo, ov, cv0 = _split_amplitudes(xy[0], delta_s, nc, no, nv)
        cv = c.T @ cv @ v
        if delta_s == 1:
            x = cv
        else:
            co, ov = c.T @ co @ o, o.T @ ov @ v
            if delta_s == 0:
                x = np.concatenate((co.ravel(), cv.ravel(), [oo], ov.ravel(), (c.T @ cv0 @ v).ravel()))
            else:
                x = np.block([[co, cv], [o.T @ oo @ o, ov]])
        rotated.xy.append((x, 0))
    actual = rotated.NAC().kernel(1, 2, atmlst=[0], ediff=True)
    np.testing.assert_allclose(actual, expected, atol=2e-8, rtol=0)
    np.testing.assert_allclose(np.abs(overlap_matrix(td, rotated)), np.eye(len(td.xy)), atol=2e-13, rtol=0)


def test_memory_propagation_and_failed_orbital_response(references):
    td = make_td(references(), 0)
    td.max_memory = 321
    nac = td.NAC()
    assert nac.max_memory == 321
    grad = td.Gradients()
    nac.max_memory = 123
    with patch.object(td, 'Gradients', return_value=grad):
        nac.kernel(1, 2, atmlst=[0])
    assert grad.max_memory == 123
    nac.cphf_max_cycle = 1
    with pytest.raises(RuntimeError):
        nac.kernel(1, 2, atmlst=[0])


def test_overlap_root_matching_handles_permutations_and_rejects_loss(references):
    td = make_td(references(), 0)
    changed = copy.copy(td)
    permutation = [2, 0, 1]
    changed.xy = [(-td.xy[k][0], 0) for k in permutation]
    changed.e = td.e[permutation]
    changed.converged = td.converged[permutation]
    matched = _match_roots(td, changed, (0, 1), 0.5)
    for original, aligned in zip(td.xy, matched):
        np.testing.assert_allclose(original[0], aligned[0], atol=1e-14)
    with pytest.raises(RuntimeError, match='state-tracking'):
        _match_roots(td, changed, (0, 1), 1.1)
    changed.converged = [False] * 3
    with pytest.raises(RuntimeError, match='did not converge'):
        _match_roots(td, changed, (0, 1), 0.5)


def test_closed_shell_raising_nac():
    mol = gto.M(atom='Li 0 0 0; H 0 0 1.6', basis='sto-3g', spin=0, unit='Bohr', verbose=0)
    mf = dft.ROKS(mol).set(xc='HF', conv_tol=1e-14, conv_tol_grad=1e-11).run()
    td = make_td(mf, 1)
    # Roots 2/3 are a degenerate pi pair here; couple two sigma states instead.
    nac = td.NAC()
    analytic = nac.kernel(1, 4, atmlst=[0], ediff=True)
    np.testing.assert_allclose(analytic, nac.finite_difference(1, 4, atmlst=[0]), atol=3e-5, rtol=0)
    np.testing.assert_allclose(overlap_matrix(td, td), np.eye(len(td.xy)), atol=2e-13)


@pytest.mark.parametrize(
    'atom,spin,charge,delta_s,average',
    [
        ('Li 0 0 0; H 0 0 3.0', 1, 1, 0, True),
        ('H 0 0 0; H 0 0 1.6; H 0 1.2 0.6', 3, 0, -1, False),
    ],
)
def test_doublet_reference_and_empty_closed_virtual_spaces(atom, spin, charge, delta_s, average):
    mol = gto.M(atom=atom, basis='sto-3g', spin=spin, charge=charge, unit='Bohr', verbose=0)
    mf = dft.ROKS(mol)
    if average:
        mf = mf.average_occ()
    mf.set(xc='HF', conv_tol=1e-14, conv_tol_grad=1e-11, max_cycle=200).run()
    td = make_td(mf, delta_s)
    nac = td.NAC()
    analytic = nac.kernel(1, 2, atmlst=[0], ediff=True)
    np.testing.assert_allclose(analytic, nac.finite_difference(1, 2, atmlst=[0]), atol=1e-6, rtol=0)
    assert np.linalg.norm(analytic) > 1e-3
