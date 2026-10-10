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

"""NTTDA derivative couplings for deltaS=-1, 0 and +1.

Polarization of the excitation gradient gives the amplitude contribution.
The interstate density supplies the AO and orbital parts of the moving-CSF
connection. See ``docs/derivations/nttda_nac.md`` for signs and conventions.
"""

import numpy as np
from pyscf import dft
from pyscf.grad import rhf as rhf_grad

from nest.aocscf import AverageOccupationROKS, SymAdaptedAverageOccupationROKS
from nest.grad.nttda import driver, ensemble, roks
from nest.nttda.nttda import _orbital_indices, transition_dm


def _check_residual(residual, tolerance):
    # Also check the assembled adjoint, independently of the solver's action.
    if not np.isfinite(residual) or residual > max(1e-11, 10 * tolerance):
        raise RuntimeError('NTTDA NAC orbital response did not converge (residual %.3e)' % residual)


def _atom_list(mol, atmlst):
    atoms = tuple(range(mol.natm) if atmlst is None else atmlst)
    if any(not isinstance(atom, (int, np.integer)) or not 0 <= atom < mol.natm for atom in atoms):
        raise ValueError('atmlst must contain atom indices in [0, %d)' % mol.natm)
    return atoms


def get_hf_interstate_numerator(nac, xy_i, xy_j, atmlst=None):
    """Return X_I.T A^[R] X_J by polarizing the quadratic gradient.

    The sums/differences must NOT be normalized. Reference-energy and
    nuclear-repulsion gradients do not enter this interstate derivative.
    """
    gradient = nac.base.Gradients().set(
        verbose=nac.verbose,
        max_memory=nac.max_memory,
        cphf_conv_tol=nac.cphf_conv_tol,
        cphf_max_cycle=nac.cphf_max_cycle,
    )
    plus = gradient.grad_elec((xy_i[0] + xy_j[0], 0), atmlst=atmlst)
    _check_residual(gradient.nttda_details.residual, nac.cphf_conv_tol)
    minus = gradient.grad_elec((xy_i[0] - xy_j[0], 0), atmlst=atmlst)
    _check_residual(gradient.nttda_details.residual, nac.cphf_conv_tol)
    return 0.25 * (plus - minus)


def _ao_connection(tdobj, gamma, atoms):
    """Contract gamma with the antisymmetric, one-sided AO derivative."""
    mo = tdobj._scf.mo_coeff
    density = mo @ gamma @ mo.T
    # get_ovlp is the BRA derivative <d(mu)/dR|nu> on the atom's AO rows.
    overlap = rhf_grad.Gradients(tdobj._scf).get_ovlp(tdobj.mol)
    result = np.zeros((len(atoms), 3))
    offsets = tdobj.mol.aoslice_by_atom()
    for k, atom in enumerate(atoms):
        p0, p1 = offsets[atom, 2:]
        result[k] = 0.5 * np.einsum('xpq,pq->x', overlap[:, p0:p1], density.T[p0:p1] - density[p0:p1])
    return result


def _orbital_connection(nac, gamma, atoms):
    """Contract the antisymmetric density with relaxed orbital rotations."""
    tdobj = nac.base
    is_ensemble = isinstance(tdobj._scf, (AverageOccupationROKS, SymAdaptedAverageOccupationROKS))
    response = ensemble if is_ensemble else roks
    source = 0.5 * (gamma - gamma.T)
    pairs = response.canonical_pairs(tdobj)
    rhs = roks.pack_m_matrix(source, pairs)
    if np.max(np.abs(rhs), initial=0.0) < 1e-15:
        return np.zeros((len(atoms), 3)), 0.0
    action, pairs = response.make_hessian_transpose_action(tdobj, pairs)
    zvector = response.solve_zvector(
        action,
        pairs,
        tdobj,
        rhs,
        tolerance=nac.cphf_conv_tol,
        max_cycle=nac.cphf_max_cycle,
    )
    adjoint = response.zvector_adjoint_matrix(tdobj, pairs, zvector)
    residual = float(np.max(np.abs(roks.pack_m_matrix(adjoint, pairs) - rhs)))
    _check_residual(residual, nac.cphf_conv_tol)
    probe_alpha, probe_beta = response.zvector_probe_densities(tdobj, pairs, zvector)
    fock = driver.reference_fock_derivative(nac, tdobj, probe_alpha, probe_beta, atoms)
    orbital = driver._orbital_gradient(tdobj, source, adjoint, fock, atmlst=atoms)
    return orbital, residual


class NonAdiabaticCouplings(rhf_grad.GradientsBase):
    """Couple two distinct, one-based roots of a single NTTDA spin channel.

    ``ediff=False`` returns (omega_J-omega_I)*d_IJ in Eh/Bohr;
    ``ediff=True`` returns d_IJ in 1/Bohr. The default is the full
    spin-adapted auxiliary-wavefunction derivative coupling.

    ``use_etfs=True`` removes the antisymmetric AO connection, retaining
    orbital relaxation. DFT grid response is omitted, as in the gradients.
    """

    _keys = rhf_grad.GradientsBase._keys | {
        'state_I',
        'state_J',
        'ediff',
        'use_etfs',
        'gap_tol',
        'cphf_conv_tol',
        'cphf_max_cycle',
        'fixed_grid',
        'step',
        'root_overlap_tol',
    }

    def __init__(self, tdobj):
        super().__init__(tdobj)
        self.max_memory = tdobj.max_memory
        self.state_I = None
        self.state_J = None
        self.ediff = False
        self.use_etfs = False
        self.gap_tol = 1e-10
        self.cphf_conv_tol = 1e-12
        self.cphf_max_cycle = None
        self.fixed_grid = isinstance(tdobj._scf, dft.KohnShamDFT)
        self.step = 2e-4
        self.root_overlap_tol = 0.5
        self.nttda_nac_details = None
        self.nac = None

    def _validate_states(self, state_i, state_j):
        tdobj = self.base
        if tdobj.deltaS not in (-1, 0, 1):
            raise ValueError('deltaS must be -1, 0, or 1')
        nopen = len(_orbital_indices(tdobj)[1])
        if (tdobj.deltaS == -1 and nopen < 2) or (tdobj.deltaS == 0 and nopen == 0):
            raise ValueError('reference spin is too small for this NTTDA channel')
        if tdobj.xy is None:
            tdobj.run()
        for name, state in (('state_I', state_i), ('state_J', state_j)):
            if not isinstance(state, (int, np.integer)) or not 1 <= state <= len(tdobj.xy):
                raise ValueError('%s must be an integer in [1, %d]' % (name, len(tdobj.xy)))
        if state_i == state_j:
            raise ValueError('NTTDA NAC requires two distinct roots')
        if not tdobj._scf.converged or not all(tdobj.converged[k - 1] for k in (state_i, state_j)):
            raise RuntimeError('converge the SCF reference and both NTTDA roots before computing a NAC')
        if not np.isrealobj(tdobj._scf.mo_coeff) or any(
            not np.isrealobj(tdobj.xy[k - 1][0]) for k in (state_i, state_j)
        ):
            raise NotImplementedError('NTTDA NAC requires real orbitals and amplitudes')
        for attribute, label in (
            ('with_df', 'Density-fitted'),
            ('with_x2c', 'X2C'),
            ('with_solvent', 'Solvent-response'),
        ):
            if getattr(tdobj._scf, attribute, None) is not None:
                raise NotImplementedError('%s NTTDA NACs are not implemented' % label)
        if tdobj._scf.do_nlc():
            raise NotImplementedError('NLC NTTDA NACs are not implemented')

    def kernel(self, state_I=None, state_J=None, atmlst=None, ediff=None, use_etfs=None):
        """Evaluate the pair in the phases of the currently stored TD roots."""
        self.mol = self.base.mol
        if state_I is not None:
            self.state_I = state_I
        if state_J is not None:
            self.state_J = state_J
        if ediff is not None:
            self.ediff = ediff
        if use_etfs is not None:
            self.use_etfs = use_etfs
        if atmlst is not None:
            self.atmlst = _atom_list(self.mol, atmlst)
        atoms = _atom_list(self.mol, self.atmlst)
        self._validate_states(self.state_I, self.state_J)
        i, j = self.state_I - 1, self.state_J - 1
        gap = float(self.base.e[j] - self.base.e[i])
        if self.ediff and abs(gap) < self.gap_tol:
            raise ZeroDivisionError('NTTDA gap %.3e is below gap_tol %.3e' % (gap, self.gap_tol))

        hf = get_hf_interstate_numerator(self, self.base.xy[i], self.base.xy[j], atoms)
        gamma = transition_dm(self.base, i, j)
        orbital, residual = _orbital_connection(self, gamma, atoms)
        ao = _ao_connection(self.base, gamma, atoms)
        csf = orbital if self.use_etfs else orbital + ao
        numerator = hf + gap * csf
        self.nttda_nac_details = {
            'hf_numerator': hf,
            'csf_ao': ao,
            'csf_orbital': orbital,
            'csf': csf,
            'csf_residual': residual,
            'full_numerator': hf + gap * (orbital + ao),
            'numerator': numerator,
            'gap': gap,
            'state_I': self.state_I,
            'state_J': self.state_J,
            'use_etfs': self.use_etfs,
        }
        self.nac = numerator / gap if self.ediff else numerator
        return self.nac

    def finite_difference(self, state_I=None, state_J=None, atmlst=None, step=None):
        """Full AWF-overlap derivative coupling in 1/Bohr for small systems.

        This independent check always includes the moving basis, regardless
        of ``ediff``/``use_etfs``. Roots are matched by cross-geometry overlap.
        """
        from ._nttda_overlap import finite_difference

        self.mol = self.base.mol
        i = self.state_I if state_I is None else state_I
        j = self.state_J if state_J is None else state_J
        self._validate_states(i, j)
        atoms = self.atmlst if atmlst is None else atmlst
        atoms = _atom_list(self.mol, atoms)
        step = self.step if step is None else step
        if not np.isfinite(step) or step <= 0:
            raise ValueError('finite-difference step must be positive and finite')
        return finite_difference(self, i - 1, j - 1, atoms, step)

    def as_scanner(self):
        # A gradient scanner would add a reference energy and does not track
        # pairs. Fail explicitly rather than inherit that incompatible API.
        raise NotImplementedError('NTTDA NAC scanners are not implemented; recompute TD roots and call kernel')


NAC = NonAdiabaticCouplings

__all__ = ['NAC', 'NonAdiabaticCouplings', 'get_hf_interstate_numerator']
