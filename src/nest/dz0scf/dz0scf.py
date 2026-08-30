import numpy as np

from pyscf import dft,  lib
from pyscf.dft import uks

def _as_spin_unpolarized_dm(dm):
    arr = np.asarray(dm)
    if arr.ndim == 2:
        dm0 = arr
    elif arr.ndim == 3 and arr.shape[0] == 2:
        dm0 = arr[0] + arr[1]
    else:
        raise ValueError(
            f'Expected a 2-D density or two spin densities; got {arr.shape}'
        )

    dm_ens = np.asarray((0.5 * dm0, 0.5 * dm0))

    mo_coeff = getattr(dm, 'mo_coeff', None)
    mo_occ = getattr(dm, 'mo_occ', None)
    if mo_coeff is not None and mo_occ is not None:
        coeff = mo_coeff
        if isinstance(coeff, (tuple, list)) or np.asarray(coeff).ndim == 3:
            coeff = coeff[0]

        occ = np.asarray(mo_occ)
        if occ.ndim == 2 and occ.shape[0] == 2:
            occ = occ[0] + occ[1]

        dm_ens = lib.tag_array(
            dm_ens,
            mo_coeff=(coeff, coeff),
            mo_occ=(0.5 * occ, 0.5 * occ),
        )

    return dm_ens

def evaluate_high_spin_energy(mf):
    evaluator = dft.ROKS(mf.mol)
    evaluator.xc = mf.xc
    evaluator.max_memory = mf.max_memory

    evaluator.grids = mf.grids
    if hasattr(mf, 'nlcgrids'):
        evaluator.nlcgrids = mf.nlcgrids

    dm_hs = evaluator.make_rdm1(mf.mo_coeff, mf.mo_occ)
    hcore = evaluator.get_hcore()
    veff = evaluator.get_veff(mf.mol, dm_hs)

    return evaluator.energy_tot(
        dm=dm_hs,
        h1e=hcore,
        vhf=veff,
    )

class _DZ0VeffMixin:
    reference_energy_semantics = 'high_spin_roks_energy_on_dz0_orbitals'
    reference_energy_stationary = False

    def get_veff(
        self,
        mol=None,
        dm=None,
        dm_last=0,
        vhf_last=0,
        hermi=1,
    ):
        if mol is None:
            mol = self.mol
        if dm is None:
            dm = self.make_rdm1()

        dm_ens = _as_spin_unpolarized_dm(dm)

        if np.ndim(dm_last) >= 2:
            dm_last = _as_spin_unpolarized_dm(dm_last)

        return uks.get_veff(
            self,
            mol,
            dm_ens,
            dm_last,
            vhf_last,
            hermi,
        )
    def high_spin_energy(self):
        return evaluate_high_spin_energy(self)

    def reference_energy(self):
        return self.high_spin_energy()

    def nuc_grad_method(self):
        """Return the Dz0SCF analytic nuclear-gradient driver."""
        from nest.grad.dz0scf import Gradients
        return Gradients(self)

class EnsembleROKS(_DZ0VeffMixin, dft.roks.ROKS):
    pass

class SymAdaptedEnsembleROKS(_DZ0VeffMixin, dft.rks_symm.SymAdaptedROKS):
    pass

def DZ0SCF(mol, xc=None):
    if mol.symmetry:
        mf = SymAdaptedEnsembleROKS(mol)
    else:
        mf = EnsembleROKS(mol)

    if xc is not None:
        mf.xc = xc

    return mf

