"""Average-occupation restricted ensemble Kohn--Sham references."""

import numpy as np

from pyscf import dft
from pyscf.lib import logger
from pyscf.scf import hf


class EnsembleRKS(dft.rks.RKS):
    """RKS with fixed ``2/1/0`` occupations and zero spin density."""

    is_ensemble_rks = True
    _keys = dft.rks.RKS._keys | {'nopen', 'reference_semantics'}

    def __init__(self, mol, xc='LDA,VWN', nopen=None, reference_semantics='average'):
        super().__init__(mol, xc=xc)
        if reference_semantics not in ('average', 'roks'):
            raise ValueError(
                "reference_semantics must be 'average' or 'roks', got %r"
                % (reference_semantics,))
        self.reference_semantics = reference_semantics
        if nopen is None:
            nopen = mol.spin
        if isinstance(nopen, bool) or int(nopen) != nopen:
            raise ValueError('nopen must be a non-negative integer')
        self.nopen = int(nopen)
        self._validate_ensemble()

    @property
    def reference_energy_semantics(self):
        if self.reference_semantics == 'roks':
            return 'roks_energy_on_ensemble_rks_orbitals'
        return 'average_occupation_ensemble_rks_energy'

    @property
    def reference_energy_stationary(self):
        return self.reference_semantics == 'average'

    def _validate_ensemble(self):
        if self.nopen < 0:
            raise ValueError('nopen must be a non-negative integer')
        if self.nopen != self.mol.spin:
            raise ValueError('nopen must match mol.spin for NTTDA')
        if self.nopen > self.mol.nelectron:
            raise ValueError('nopen cannot exceed the electron count')
        if (self.mol.nelectron - self.nopen) % 2:
            raise ValueError('electron count and nopen have inconsistent parity')

    @property
    def nclosed(self):
        return (self.mol.nelectron - self.nopen) // 2

    def check_sanity(self):
        out = hf.SCF.check_sanity(self)
        if self.do_nlc() and self.do_disp() and self._numint.libxc.is_nlc(self.xc):
            import warnings
            warnings.warn(
                f'nlc-type xc {self.xc} and disp {self.disp} may lead to double counting in NLC.'
            )
        return out

    def get_occ(self, mo_energy=None, mo_coeff=None):
        self._validate_ensemble()
        if mo_energy is None:
            mo_energy = self.mo_energy
        mo_energy = np.asarray(mo_energy)
        if self.nclosed + self.nopen > mo_energy.size:
            raise RuntimeError('not enough orbitals for the requested ensemble occupations')

        order = np.argsort(mo_energy, kind='stable')
        mo_occ = np.zeros_like(mo_energy)
        mo_occ[order[:self.nclosed]] = 2
        mo_occ[order[self.nclosed:self.nclosed + self.nopen]] = 1
        if self.verbose >= logger.INFO:
            logger.info(self, 'EnsembleRKS occupations = %s', mo_occ)
        return mo_occ

    def make_rdm1s(self, mo_coeff=None, mo_occ=None):
        dm = self.make_rdm1(mo_coeff, mo_occ)
        dm_spin = np.asarray(dm) * 0.5
        return dm_spin, dm_spin.copy()

    def reference_energy(self):
        if self.reference_semantics == 'roks':
            if self.mo_coeff is None or self.mo_occ is None:
                raise RuntimeError(
                    'run EnsembleRKS.kernel() before evaluating the reference energy')
            evaluator = dft.ROKS(self.mol).set(
                xc=self.xc,
                nlc=self.nlc,
                max_memory=self.max_memory,
                verbose=0,
            )
            evaluator.grids = self.grids
            evaluator.nlcgrids = self.nlcgrids
            dm = evaluator.make_rdm1(self.mo_coeff, self.mo_occ)
            hcore = evaluator.get_hcore(self.mol)
            veff = evaluator.get_veff(self.mol, dm)
            return float(evaluator.energy_tot(dm=dm, h1e=hcore, vhf=veff))
        if self.e_tot is None:
            raise RuntimeError('run EnsembleRKS.kernel() before evaluating the reference energy')
        return float(self.e_tot)

    def nuc_grad_method(self):
        """Return the analytic gradient driver of the selected reference energy.

        For the ``average`` semantics this is the gradient of the stationary
        ensemble energy (the stock RKS driver).  For the ``roks`` semantics it
        differentiates the fixed-orbital ROKS energy returned by
        :meth:`reference_energy`, including its Z-vector orbital-relaxation
        contribution and the nuclear-repulsion gradient.
        """
        if self.reference_semantics == 'roks':
            from nest.grad.nttda.reference import ReferenceGradients
            return ReferenceGradients(self)
        return super().nuc_grad_method()

    def get_grad(self, mo_coeff, mo_occ, fock=None):
        mo_occ = np.asarray(mo_occ)
        if fock is None:
            dm = self.make_rdm1(mo_coeff, mo_occ)
            fock = self.get_hcore(self.mol) + self.get_veff(self.mol, dm)
        fock_mo = mo_coeff.conj().T @ fock @ mo_coeff
        unique = hf.uniq_var_indices(mo_occ)
        occupation_difference = mo_occ[None, :] - mo_occ[:, None]
        return (fock_mo * occupation_difference)[unique]
