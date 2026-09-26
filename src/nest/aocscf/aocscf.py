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


def _high_spin_energy_and_summary(mf):
    """Evaluate high-spin energy with an independent energy summary."""
    if mf.mo_coeff is None or mf.mo_occ is None:
        raise RuntimeError('Run SCF before evaluating the high-spin energy.')
    # A view bypasses the spin-averaged get_veff while retaining the original
    # object's XC integrator, grids, and other calculation settings.
    mf_hs = lib.view(mf, dft.roks.ROKS)
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
    nuc_grad_method = nuc_grad_method
    Gradients = nuc_grad_method
    dump_flags = dump_flags
    check_sanity = check_sanity


def average_occ(mf):
    """Return an average-occupation SCF object from a plain PySCF ROKS."""
    if type(mf) is dft.roks.ROKS:
        cls = AverageOccupationROKS
    elif type(mf) is dft.rks_symm.SymAdaptedROKS:
        cls = SymAdaptedAverageOccupationROKS
    else:
        raise TypeError('average_occ() requires a plain ROKS or SymAdaptedROKS object')
    averaged = lib.view(mf, cls)
    averaged.scf_summary = mf.scf_summary.copy()
    averaged.converged = False
    averaged.e_tot = None
    averaged.e_avg_occ = None
    return averaged


dft.roks.ROKS.average_occ = average_occ
dft.rks_symm.SymAdaptedROKS.average_occ = average_occ
