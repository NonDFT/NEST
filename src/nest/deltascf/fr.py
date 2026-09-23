#!/usr/bin/env python
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
#
# Ref: JCTC 2027, 22, 4609

"""First-order freeze-and-release SCF for restricted two-SOMO triplets.

ROHF/ROKS adaptation of Qin and Suo's FR-TO delta-SCF workflow. The
frozen stage uses exact complementary-space diagonalization, rather than
finite projector shifts in the paper's unrestricted spin channel.
"""

import weakref

import numpy as np
from scipy.linalg import eigh
from pyscf import lib, scf, symm
from pyscf.lib import logger
from pyscf.scf import hf


def _set_imom(mf, mo_ref, setocc):
    # Both occupation closures need access to mf, but must not own it.
    mf = weakref.proxy(mf)
    scf.addons.mom_occ(mf, mo_ref, setocc)
    imom_occ = mf.get_occ
    if mf.mol.symmetry:
        reference_symmetry = mf.get_orbsym(mo_ref)
        overlap = mf.get_ovlp()

    def get_occ(mo_energy=None, mo_coeff=None):
        if not mf.mol.symmetry:
            occupations = imom_occ(mo_energy, mo_coeff)
        else:
            # Preserve the reference spin occupations within each irrep.
            if mo_coeff is None:
                mo_coeff = mf.mo_coeff
            orbital_symmetry = mf.get_orbsym(mo_coeff)
            spin_occupations = np.zeros((2, mo_coeff.shape[1]))
            for irrep in np.unique(reference_symmetry):
                candidates = np.flatnonzero(orbital_symmetry == irrep)
                for spin in range(2):
                    occupied_reference = (reference_symmetry == irrep) & (setocc[spin] > 0)
                    noccupied = np.count_nonzero(occupied_reference)
                    if len(candidates) < noccupied:
                        raise ValueError('Not enough orbitals in an IMOM symmetry block')
                    overlaps = mo_ref[:, occupied_reference].T @ overlap @ mo_coeff[:, candidates]
                    weights = np.sum(overlaps**2, axis=0)
                    selected = candidates[np.argsort(-weights, kind='stable')[:noccupied]]
                    spin_occupations[spin, selected] = 1
            occupations = spin_occupations.sum(axis=0)
        # Independent alpha/beta rankings can violate restricted nesting.
        if np.count_nonzero(occupations == 1) != 2:
            raise RuntimeError('IMOM selected nonnested alpha/beta spaces incompatible with a ROHF triplet')
        return occupations

    mf.get_occ = get_occ


class FR(lib.StreamObject):
    """Freeze both SOMOs, then release all orbitals with IMOM (ROHF/ROKS).

    Requires spin=2 and real S-orthonormal input orbitals. With symmetry,
    each orbital must have a definite irrep in an Abelian point group;
    multiple orbitals may share an irrep. Dooh, Coov and SO3 are unsupported.

    Attributes:
        freeze_tol : float
            Frozen-stage orbital-gradient threshold. Default is 1e-5.
        freeze_max_cycle : int
            Maximum frozen-stage iterations. Default is 100.
            Failure to converge raises before release starts.
        conv_tol, conv_tol_grad : float
            Energy and gradient thresholds for release; inherited from mf.
        max_cycle : int
            Maximum release iterations; inherited from mf.
        callback : callable
            Called after each iteration with the PySCF environment dict.
            env['fr_stage'] is 'freeze' or 'release'. Default is None.

    Saved results:
        freeze_converged : bool
            Whether the frozen stage converged.
        freeze_cycles : int
            Number of frozen-stage iterations.
        converged, cycles
            Convergence status and iteration count of the release stage.
        e_tot, mo_energy, mo_coeff, mo_occ
            Final energy, orbital energies, coefficients and occupations.

    Examples:
        >>> excited = mf.FR()
        >>> excited.kernel(mo_coeff, mo_occ)
    """

    __name_mixin__ = 'FR'
    freeze_tol = 1e-5
    freeze_max_cycle = 100
    freeze_cycles = 0
    freeze_converged = None
    _keys = {'freeze_tol', 'freeze_max_cycle', 'freeze_cycles', 'freeze_converged'}

    def __new__(cls, mf):
        if isinstance(mf, FR):
            return mf
        if not isinstance(mf, hf.SCF) or not mf.istype('ROHF') or hasattr(mf, '_scf'):
            raise NotImplementedError('FR requires a plain ROHF/ROKS object')
        return lib.set_class(object.__new__(cls), (cls, mf.__class__))

    def __init__(self, mf):
        if mf is not self:
            self.__dict__.update(mf.__dict__)

    def check_sanity(self):
        if self.mol.spin != 2:
            raise ValueError('FR requires spin=2')
        if self.mol.symmetry and self.mol.groupname in ('Dooh', 'Coov', 'SO3'):
            raise NotImplementedError('FR supports Abelian point groups; select an Abelian subgroup')
        if not np.isfinite(self.freeze_tol) or self.freeze_tol <= 0:
            raise ValueError('freeze_tol must be finite and positive')
        if not isinstance(self.freeze_max_cycle, (int, np.integer)) or self.freeze_max_cycle < 1:
            raise ValueError('freeze_max_cycle must be a positive integer')
        super().check_sanity()
        return self

    def dump_flags(self, verbose=None):
        super().dump_flags(verbose)
        log = logger.new_logger(self, verbose)
        log.info('FR freeze gradient tolerance = %g', self.freeze_tol)
        log.info('FR freeze max_cycle = %d', self.freeze_max_cycle)
        log.info('FR release occupation method = IMOM')
        return self

    def kernel(self, mo_coeff=None, mo_occ=None):
        self.check_sanity()
        if self.verbose >= logger.INFO:
            self.dump_flags()
        log = logger.new_logger(self)

        c = np.array(self.mo_coeff if mo_coeff is None else mo_coeff, copy=True)
        occ = np.array(self.mo_occ if mo_occ is None else mo_occ, copy=True)
        if (occ.ndim != 1 or
                not np.all(np.isin(occ, [0, 1, 2])) or
                np.count_nonzero(occ == 1) != 2 or occ.sum() != self.mol.nelectron):
            raise ValueError('FR requires a spin=2 occupation vector with exactly two SOMOs')
        if c.ndim != 2 or c.shape != (self.mol.nao_nr(), occ.size) or np.iscomplexobj(c):
            raise ValueError('FR requires real MO coefficients matching mo_occ')
        s = self.get_ovlp()
        if not np.allclose(c.T @ s @ c, np.eye(occ.size), atol=1e-8, rtol=0):
            raise ValueError('FR requires S-orthonormal input orbitals')
        orbital_symmetry = np.zeros(occ.size, dtype=int)
        if self.mol.symmetry:
            orbital_symmetry = symm.label_orb_symm(
                self.mol, self.mol.irrep_id, self.mol.symm_orb, c, s=s, check=True)
            c = lib.tag_array(c, orbsym=orbital_symmetry)
            for name, irrep in zip(self.mol.irrep_name, self.mol.irrep_id):
                alpha = np.count_nonzero((orbital_symmetry == irrep) & (occ > 0))
                beta = np.count_nonzero((orbital_symmetry == irrep) & (occ == 2))
                requested = self.irrep_nelec.get(name)
                if requested is not None:
                    matches = (requested == alpha + beta if np.isscalar(requested)
                               else tuple(requested) == (alpha, beta))
                    if not matches:
                        raise ValueError('Input occupations conflict with irrep_nelec for ' + name)

        self.converged = False
        self.freeze_converged = False
        self.freeze_cycles = 0
        # Drop only this mixin: the two stages retain all physical MF settings.
        base = lib.view(self, lib.drop_class(self.__class__, FR)).copy()
        for key in FR._keys:
            base.__dict__.pop(key, None)
        frozen, release = base.copy(), base.copy()
        callback = self.callback
        for stage, mf in [('freeze', frozen), ('release', release)]:
            mf.conv_check = False
            if self.diis:
                if stage == 'freeze':
                    mf.diis = scf.diis.CDIIS(mf)
                    mf.diis.space = self.diis_space
                    mf.diis.rollback = self.diis_space_rollback
                    mf.diis.damp = self.diis_damp
                else:
                    # Start release with a fresh DIIS history.
                    mf.diis = True
                    mf.DIIS = scf.diis.CDIIS
            def stage_callback(env, stage=stage):
                if callback is not None:
                    callback(dict(env, fr_stage=stage))
            mf.callback = stage_callback

        # Exclude SOMOs; place doubly occupied slots before virtual slots.
        doubly_occupied_indices = np.flatnonzero(occ == 2)
        virtual_indices = np.flatnonzero(occ == 0)
        free_indices = np.concatenate((doubly_occupied_indices, virtual_indices))
        free_orbitals = c[:, free_indices]
        free_blocks = [free_indices[orbital_symmetry[free_indices] == irrep]
                       for irrep in np.unique(orbital_symmetry[free_indices])]
        # Bind to the unmodified base, not the object that owns this closure.
        original_get_grad = base.get_grad

        def frozen_eig(fock, overlap, **kwargs):
            # B.T @ S @ B = I; diagonalize each irrep separately.
            coeff = c.copy()
            energies = np.einsum('pi,pi->i', c, fock @ c)
            for indices in free_blocks:
                basis = c[:, indices]
                eigenvalues, rotation = eigh(basis.T @ fock @ basis)
                coeff[:, indices] = basis @ rotation
                energies[indices] = eigenvalues
            if self.mol.symmetry:
                coeff = lib.tag_array(coeff, orbsym=orbital_symmetry)
            return energies, coeff

        def frozen_get_occ(mo_energy=None, mo_coeff=None):
            # frozen_eig already orders the free orbitals by energy.
            return occ.copy()

        def frozen_get_grad(coeff, occupations, fock=None):
            # Select doubly occupied -> virtual entries in PySCF's packed gradient.
            # Rebuild the mask because symmetry finalization can reorder orbitals.
            alpha_rotations = (occupations == 0)[:, None] & (occupations > 0)[None, :]
            beta_rotations = (occupations != 2)[:, None] & (occupations == 2)[None, :]
            all_rotations = alpha_rotations | beta_rotations
            doubly_to_virtual = (occupations == 0)[:, None] & (occupations == 2)[None, :]
            free_gradient_entries = doubly_to_virtual[all_rotations]
            gradient = original_get_grad(coeff, occupations, fock)
            return gradient[free_gradient_entries]

        frozen.eig = frozen_eig
        frozen.get_occ = frozen_get_occ
        frozen.get_grad = frozen_get_grad
        if frozen.diis:
            # Project the DIIS residual onto the free orbital space.
            if self.mol.symmetry:
                free_orbitals = lib.tag_array(free_orbitals, orbsym=orbital_symmetry[free_indices])
            frozen.diis.Corth = free_orbitals
        frozen.conv_tol_grad = self.freeze_tol
        frozen.max_cycle = self.freeze_max_cycle
        log.info('FR Freeze stage begins: fix both singly occupied orbitals; '
                    'relax doubly occupied/virtual orbitals')
        frozen.kernel(dm0=frozen.make_rdm1(c, occ))
        self.freeze_cycles = frozen.cycles
        self.freeze_converged = frozen.converged
        if not self.freeze_converged:
            raise RuntimeError('FR frozen SCF did not converge; release was not started')

        log.info('FR Release stage begins: relax all orbitals with the post-freeze IMOM reference')
        # Use the post-freeze reference, including symmetry reordering.
        setocc = np.asarray((frozen.mo_occ > 0, frozen.mo_occ == 2), dtype=float)
        _set_imom(release, frozen.mo_coeff, setocc)
        release.kernel(dm0=frozen.make_rdm1())
        # Copy results without retaining stage-local hooks.
        for key, value in release.__dict__.items():
            if key not in ('callback', 'get_occ', 'diis', 'DIIS'):
                self.__dict__[key] = value
        _set_imom(self, frozen.mo_coeff, setocc)
        somo_overlap = c[:, occ == 1].T @ s @ self.mo_coeff[:, self.mo_occ == 1]
        log.note('FR target SOMO overlap singular values: %s',
                    np.linalg.svd(somo_overlap, compute_uv=False))
        return self.e_tot

    scf = kernel


hf.SCF.FR = lib.class_as_method(FR)
