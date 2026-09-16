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

import numpy as np
from scipy.linalg import eigh
from pyscf import lib, scf, symm
from pyscf.lib import logger
from pyscf.scf import hf


def _set_imom(mf, mo_ref, setocc):
    scf.addons.mom_occ(mf, mo_ref, setocc)
    imom_occ = mf.get_occ
    if mf.mol.symmetry:
        reference_symmetry = mf.get_orbsym(mo_ref)
        overlap = mf.get_ovlp()

    def get_occ(mo_energy=None, mo_coeff=None):
        if not mf.mol.symmetry:
            occupations = imom_occ(mo_energy, mo_coeff)
        else:
            # PySCF mom_occ ranks all orbitals together. Here retain the
            # reference alpha/beta electron counts within each irrep, so
            # overlap ranking cannot move electrons between symmetry blocks.
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
    """Freeze both singly occupied orbitals, relax the others, then run IMOM.

    Use ``mf.FR().kernel(mo_coeff, mo_occ)`` on a plain ROHF/ROKS object.
    Requires real S-orthonormal orbitals and spin=2. With symmetry enabled,
    input orbitals must belong to individual irreps of an Abelian point
    group (e.g. Cs, C2v, D2h). Both stages retain the initial alpha/beta
    electron counts in each irrep. For linear molecules or atoms, select
    an Abelian subgroup explicitly; Dooh, Coov and SO3 are not supported.
    Both stages use ordinary SCF and fresh CDIIS histories, with no Hessian.
    Energies always come from the physical Hamiltonian.

    freeze_tol (default 1e-5) bounds PySCF's projected orbital-gradient norm;
    freeze_max_cycle (100) bounds the frozen stage. Failure raises before
    release. freeze_cycles/freeze_converged describe that stage; cycles and
    converged describe release, governed by the usual SCF tolerances.
    callback receives both stages, identified by env['fr_stage'].
    PySCF's extra relaxed convergence check is disabled in both stages.
    At INFO verbosity (verbose=4), stage starts and the final SOMO-subspace
    overlap singular values relative to the input orbitals are logged.
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

    def kernel(self, mo_coeff=None, mo_occ=None):
        c = np.array(self.mo_coeff if mo_coeff is None else mo_coeff, copy=True)
        occ = np.array(self.mo_occ if mo_occ is None else mo_occ, copy=True)
        if (self.mol.spin != 2 or occ.ndim != 1 or
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
            if self.mol.groupname not in ('C1', 'Ci', 'Cs', 'C2', 'C2v', 'C2h', 'D2', 'D2h'):
                raise NotImplementedError('FR supports Abelian point groups; select an Abelian subgroup')
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
        if self.freeze_tol <= 0 or self.freeze_max_cycle < 1:
            raise ValueError('freeze_tol and freeze_max_cycle must be positive')

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
                    # Let the ordinary kernel build its orthogonalized DIIS
                    # residual, without carrying over the frozen history.
                    mf.diis = True
                    mf.DIIS = scf.diis.CDIIS
            def stage_callback(env, stage=stage):
                if callback is not None:
                    callback(dict(env, fr_stage=stage))
            mf.callback = stage_callback

        # The two singly occupied columns never change during the freeze
        # iterations. All other orbitals are linear combinations of the
        # INITIAL doubly occupied and virtual columns, an S-orthonormal basis
        # for the allowed space. Put occupied indices first: eigh returns
        # increasing energies, so the lowest eigenvectors fill these slots.
        doubly_occupied_indices = np.flatnonzero(occ == 2)
        virtual_indices = np.flatnonzero(occ == 0)
        free_indices = np.concatenate((doubly_occupied_indices, virtual_indices))
        free_orbitals = c[:, free_indices]
        free_blocks = [free_indices[orbital_symmetry[free_indices] == irrep]
                       for irrep in np.unique(orbital_symmetry[free_indices])]
        original_get_grad = frozen.get_grad

        def frozen_eig(fock, overlap, **kwargs):
            # In each free block B, B.T @ S @ B = I. Thus this is an ordinary
            # eigenproblem, not an AO generalized eigenproblem. Separate
            # irreps must not mix, even if their eigenvalues are degenerate.
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
            # frozen_eig already placed the lowest free eigenvectors into
            # the doubly occupied slots; keep the initial occupation labels.
            return occ.copy()

        def frozen_get_grad(coeff, occupations, fock=None):
            # PySCF packs all nonredundant ROHF rotations into a 1-D vector:
            # alpha: occupied -> virtual; beta: doubly -> singly/virtual.
            # Select only doubly occupied -> virtual entries of THAT vector.
            # Derive masks from the supplied occupations because symmetry
            # SCF can reorder orbitals during its finalization step.
            alpha_rotations = (occupations == 0)[:, None] & (occupations > 0)[None, :]
            beta_rotations = (occupations != 2)[:, None] & (occupations == 2)[None, :]
            all_rotations = alpha_rotations | beta_rotations
            doubly_to_virtual = (occupations == 0)[:, None] & (occupations == 2)[None, :]
            free_gradient_entries = doubly_to_virtual[all_rotations]
            gradient = original_get_grad(coeff, occupations, fock)
            return gradient[free_gradient_entries]

        # These three hooks implement the constrained eigenproblem, filling,
        # and convergence test. Density/Fock/energy updates stay in PySCF.
        frozen.eig = frozen_eig
        frozen.get_occ = frozen_get_occ
        frozen.get_grad = frozen_get_grad
        if frozen.diis:
            # CDIIS must ignore rotations involving fixed singly occupied
            # orbitals too. For the physical ROHF Fock, its virtual/doubly
            # occupied block is (F_alpha + F_beta)/2. Multiplying by the
            # occupation difference 2 gives exactly PySCF's orbital gradient:
            # ||B.T @ (F D S - S D F) @ B||_F = sqrt(2) * ||g_free||_2,
            # where D is the total (alpha + beta) density. The same-irrep
            # restriction is applied by CDIIS using the orbsym tag below.
            if self.mol.symmetry:
                free_orbitals = lib.tag_array(free_orbitals, orbsym=orbital_symmetry[free_indices])
            frozen.diis.Corth = free_orbitals
        frozen.conv_tol_grad = self.freeze_tol
        frozen.max_cycle = self.freeze_max_cycle
        logger.info(self, 'FR Freeze stage begins: fix both singly occupied orbitals; '
                    'relax doubly occupied/virtual orbitals')
        frozen.kernel(dm0=frozen.make_rdm1(c, occ))
        self.freeze_cycles = frozen.cycles
        self.freeze_converged = frozen.converged
        if not self.freeze_converged:
            raise RuntimeError('FR frozen SCF did not converge; release was not started')

        # PySCF mom_occ captures these occupied reference spaces once (IMOM).
        # Resetting the reference after freeze incorporates spectator relaxation.
        logger.info(self, 'FR Release stage begins: relax all orbitals with the post-freeze IMOM reference')
        # Symmetry SCF finalization sorts columns AND occupations. Use both
        # returned arrays together, rather than reusing initial column labels.
        setocc = np.asarray((frozen.mo_occ > 0, frozen.mo_occ == 2), dtype=float)
        _set_imom(release, frozen.mo_coeff, setocc)
        release.kernel(dm0=frozen.make_rdm1())
        # With conv_check=False the ordinary kernel already checks energy
        # AND orbital-gradient convergence. Honor its result, including any
        # user-supplied check_convergence callback, without a second criterion.
        # Retain the ordinary MF results and caches, but not stage-local hooks.
        for key, value in release.__dict__.items():
            if key not in ('callback', 'get_occ', 'diis', 'DIIS'):
                self.__dict__[key] = value
        _set_imom(self, frozen.mo_coeff, setocc)
        # Use the final occupation labels: IMOM and symmetry finalization
        # can move the singly occupied orbitals to different column indices.
        somo_overlap = c[:, occ == 1].T @ s @ self.mo_coeff[:, self.mo_occ == 1]
        logger.info(self, 'FR target SOMO overlap singular values: %s',
                    np.linalg.svd(somo_overlap, compute_uv=False))
        return self.e_tot

    scf = kernel


hf.SCF.FR = lib.class_as_method(FR)
