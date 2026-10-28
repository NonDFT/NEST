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
#
# Author: Yue Yu & Tai Wang
#
"""ROKS orbitals optimized with spin-averaged occupations."""

import numpy as np
from pyscf import dft, lib
from pyscf.lib import logger
from pyscf.soscf import newton_ah


def get_veff(mf, mol=None, dm=None, dm_last=0, vhf_last=0, hermi=1):
    if mol is None:
        mol = mf.mol
    if dm is None:
        dm = mf.make_rdm1()
    if np.ndim(dm) == 3:
        dm = dm[0] + dm[1]
    if np.ndim(dm_last) == 3:
        dm_last = dm_last[0] + dm_last[1]
    veff = dft.rks.get_veff(mf, mol, dm, dm_last, vhf_last, hermi)
    # ROHF.get_fock expects separate alpha and beta blocks, even when equal.
    return lib.tag_array(np.asarray((veff, veff)), ecoul=veff.ecoul, exc=veff.exc, vj=veff.vj, vk=veff.vk)


def gen_response(mf, mo_coeff=None, mo_occ=None, with_j=True, hermi=0, max_memory=None, with_nlc=True):
    """Differentiate the common average-occupation potential in ROHF spin blocks."""
    if mo_coeff is None:
        mo_coeff = mf.mo_coeff
    if mo_occ is None:
        mo_occ = mf.mo_occ
    # ROHF second-order solvers supply the shared orbitals and occupations
    # as separate alpha/beta arrays. RKS needs the total occupations.
    if np.ndim(mo_coeff) == 3:
        mo_coeff = mo_coeff[0]
    if np.ndim(mo_occ) == 2:
        mo_occ = np.asarray(mo_occ).sum(axis=0)
    response_mf = lib.view(mf, dft.rks.RKS)
    # Keep the original J/K backend, including density fitting.
    response_mf.get_jk = mf.get_jk
    response_mf.get_j = mf.get_j
    response_mf.get_k = mf.get_k
    response = response_mf.gen_response(
        mo_coeff, mo_occ, hermi=hermi, max_memory=max_memory, with_nlc=with_nlc)

    def vind(dm1):
        total_density = dm1[0] + dm1[1]
        potential = response(total_density)
        # RKS includes Coulomb response except for anti-Hermitian probes.
        if not with_j and hermi != 2:
            potential -= mf.get_j(dm=total_density, hermi=hermi)
        return np.asarray((potential, potential))

    return vind


class SecondOrderAverageOccupationROKS(newton_ah._SecondOrderROHF):
    def sfx2c1e(self):
        from pyscf.x2c import sfx2c1e

        mf = sfx2c1e.sfx2c1e(self)
        # Newton evaluates the energy with _scf, and the Hessian with self.
        mf._scf = self._scf.sfx2c1e()
        mf._scf.with_x2c = mf.with_x2c
        return mf

    def gen_g_hop(self, mo_coeff, mo_occ, fock_ao=None, h1e=None, with_symmetry=True):
        # Retain the shared rotation's oo/vv terms and gradient transport,
        # which PySCF's separate-spin ROHF Hessian omits away from convergence.
        from nest.deltascf.sgm import gen_g_hop_rohf

        return gen_g_hop_rohf(self, mo_coeff, mo_occ, fock_ao, h1e, with_symmetry)


def newton(mf):
    """Use PySCF's Newton driver with the shared-orbital AOCSCF Hessian."""
    if isinstance(mf, newton_ah._CIAH_SOSCF):
        return mf
    cls = SecondOrderAverageOccupationROKS
    return lib.set_class(cls(mf), (cls, mf.__class__))


def _high_spin_energy_and_summary(mf):
    """Evaluate high-spin energy with an independent energy summary."""
    if mf.mo_coeff is None or mf.mo_occ is None:
        raise RuntimeError('Run SCF before evaluating the high-spin energy.')
    # Remove the solver and replace only AOCSCF, keeping X2C/DF and symmetry.
    mf_hs = mf.remove_soscf()
    if hasattr(mf_hs, 'undo_sgm'):
        mf_hs = mf_hs.undo_sgm()
    if isinstance(mf_hs, SymAdaptedAverageOccupationROKS):
        cls = lib.replace_class(mf_hs.__class__, SymAdaptedAverageOccupationROKS,
                                dft.rks_symm.SymAdaptedROKS)
    else:
        cls = lib.replace_class(mf_hs.__class__, AverageOccupationROKS, dft.roks.ROKS)
    mf_hs = lib.view(mf_hs, cls)
    mf_hs.scf_summary = mf.scf_summary.copy()
    dm_hs = mf_hs.make_rdm1(mf.mo_coeff, mf.mo_occ)
    veff_hs = mf_hs.get_veff(mf.mol, dm_hs)
    energy = mf_hs.energy_tot(dm=dm_hs, h1e=mf_hs.get_hcore(), vhf=veff_hs)
    return energy, mf_hs.scf_summary


def nuc_grad_method(mf):
    """Return the analytic gradient of the high-spin reported energy."""
    from nest.grad.aocscf import Gradients

    return Gradients(mf)


def dump_flags(mf, verbose=None):
    if isinstance(mf, dft.rks_symm.SymAdaptedROKS):
        dft.rks_symm.SymAdaptedROKS.dump_flags(mf, verbose)
    else:
        dft.roks.ROKS.dump_flags(mf, verbose)
    log = logger.new_logger(mf, verbose)
    log.info('SCF potential: average occupation (equal alpha/beta density)')
    log.info('Reported total energy and nuclear gradient: high spin')
    return mf


def check_sanity(mf):
    dft.roks.ROKS.check_sanity(mf)
    if mf.mo_occ is not None and np.ndim(mf.mo_occ) != 1:
        raise ValueError('Average-occupation ROKS requires one-dimensional MO occupations')
    return mf


class AverageOccupationROKS(dft.roks.ROKS):
    """ROKS with average-occupation SCF orbitals and a high-spin ``e_tot``."""

    _keys = {'e_avg_occ'}

    def _finalize(self):
        # PySCF's SCF driver needs energy_tot() to remain the average-occupation
        # energy during iterations and convergence checks.
        # The final reported energy and nuclear gradient are the high-spin values.
        # e_tot is set to the high-spin energy with `_finalize()` after SCF
        self.e_avg_occ = self.e_tot
        if self.mo_coeff is not None and self.mo_occ is not None:
            self.e_tot, self.scf_summary = _high_spin_energy_and_summary(self)
        logger.note(self, 'average-occupation SCF energy = %.15g', self.e_avg_occ)
        return dft.roks.ROKS._finalize(self)

    get_veff = get_veff
    gen_response = gen_response
    newton = newton
    nuc_grad_method = nuc_grad_method
    Gradients = nuc_grad_method
    dump_flags = dump_flags
    check_sanity = check_sanity

class SymAdaptedAverageOccupationROKS(dft.rks_symm.SymAdaptedROKS):
    """Symmetry-adapted average-occupation ROKS."""

    _keys = {'e_avg_occ'}

    def _finalize(self):
        self.e_avg_occ = self.e_tot
        if self.mo_coeff is not None and self.mo_occ is not None:
            self.e_tot, self.scf_summary = _high_spin_energy_and_summary(self)
        logger.note(self, 'average-occupation SCF energy = %.15g', self.e_avg_occ)
        return dft.rks_symm.SymAdaptedROKS._finalize(self)

    get_veff = get_veff
    gen_response = gen_response
    newton = newton
    nuc_grad_method = nuc_grad_method
    Gradients = nuc_grad_method
    dump_flags = dump_flags
    check_sanity = check_sanity


def average_occ(mf):
    """Replace the ROKS reference while retaining its X2C/DF/SCF wrappers."""
    if isinstance(mf, (AverageOccupationROKS, SymAdaptedAverageOccupationROKS)):
        return mf
    if isinstance(mf, dft.rks_symm.SymAdaptedROKS):
        cls = lib.replace_class(mf.__class__, dft.rks_symm.SymAdaptedROKS,
                                SymAdaptedAverageOccupationROKS)
    elif isinstance(mf, dft.roks.ROKS):
        cls = lib.replace_class(mf.__class__, dft.roks.ROKS, AverageOccupationROKS)
    else:
        raise TypeError('average_occ() requires a ROKS or SymAdaptedROKS object')
    if isinstance(mf, newton_ah._CIAH_SOSCF):
        cls = lib.replace_class(cls, newton_ah._SecondOrderROHF, SecondOrderAverageOccupationROKS)
    averaged = lib.view(mf, cls)
    if isinstance(mf, newton_ah._CIAH_SOSCF):
        averaged._scf = average_occ(mf._scf)
        if getattr(mf, 'with_x2c', None) is not None:
            averaged = averaged.sfx2c1e()
    averaged.scf_summary = mf.scf_summary.copy()
    averaged.converged = False
    averaged.e_tot = None
    averaged.e_avg_occ = None
    return averaged


dft.roks.ROKS.average_occ = average_occ
dft.rks_symm.SymAdaptedROKS.average_occ = average_occ
