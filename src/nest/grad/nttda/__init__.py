"""Analytic nuclear gradients for :mod:`nest.nttda`."""

import numpy as np

from pyscf import dft, lib
from pyscf.grad import rhf as rhf_grad
from pyscf.lib import logger
from nest.nttda.nttda import NTTDA

from . import delta_s_minus_one, delta_s_plus_one, delta_s_zero



def _normalized_amplitude(xy):
    vector = np.asarray(xy[0]).ravel()
    return vector / np.linalg.norm(vector)


def _copy_td_settings(source, target):
    for name in (
            "deltaS", "nobeta", "nstates", "conv_tol", "lindep",
            "max_cycle", "max_memory"):
        setattr(target, name, getattr(source, name))
    target.verbose = 0
    return target


def _displaced_reference(source, mol, fixed_grid):
    if isinstance(source, dft.KohnShamDFT):
        reference = dft.ROKS(mol)
    else:
        reference = source.__class__(mol)
    for name in (
            "conv_tol", "conv_tol_grad", "max_cycle", "max_memory",
            "level_shift", "damp"):
        if hasattr(source, name):
            setattr(reference, name, getattr(source, name))
    reference.verbose = 0
    if isinstance(source, dft.KohnShamDFT):
        reference.xc = source.xc
        reference.nlc = source.nlc
        reference.grids.level = source.grids.level
        reference.grids.prune = source.grids.prune
        if fixed_grid:
            reference.grids.coords = np.array(source.grids.coords, copy=True)
            reference.grids.weights = np.array(source.grids.weights, copy=True)
            reference.grids.non0tab = None
            reference.grids.verbose = 0
    return reference


class Gradients(rhf_grad.GradientsBase):
    """Analytic NTTDA gradients for ``deltaS=-1,0,+1``."""

    _keys = rhf_grad.GradientsBase._keys | {
        "state", "method", "step", "fixed_grid", "root_overlap_tol",
        "cphf_conv_tol", "cphf_max_cycle",
    }

    def __init__(self, tdobj):
        super().__init__(tdobj)
        self.state = 1
        self.method = "analytic"
        self.step = 1e-3
        self.fixed_grid = isinstance(tdobj._scf, dft.KohnShamDFT)
        self.root_overlap_tol = 0.5
        self.cphf_conv_tol = 1e-12
        self.cphf_max_cycle = None
        self.nttda_details = None

    def dump_flags(self, verbose=None):
        log = logger.new_logger(self, verbose)
        log.info("******** NTTDA nuclear gradients ********")
        log.info("state = %d", self.state)
        log.info("deltaS = %d", self.base.deltaS)
        log.info("nobeta = %s", self.base.nobeta)
        log.info("method = %s", self.method)
        log.info("fixed_grid = %s", self.fixed_grid)
        if self.method == "finite_diff":
            log.info("finite-difference step = %.6g Bohr", self.step)
        return self

    def grad_nuc(self, atmlst=None):
        """Ground-state ROKS gradient, including nuclear repulsion."""
        if atmlst is not None:
            atmlst = list(atmlst)
        return self.base._scf.nuc_grad_method().kernel(atmlst=atmlst)

    def _analytic_components(self, xy, atmlst):
        tdobj = self.base
        options = {
            "atmlst": atmlst,
            "tolerance": self.cphf_conv_tol,
            "max_cycle": self.cphf_max_cycle,
        }
        if tdobj.deltaS == -1:
            return delta_s_minus_one.grad_elec(
                self, tdobj, xy, **options,
            )
        if tdobj.deltaS == 0:
            return delta_s_zero.grad_elec(
                self, tdobj, xy, **options,
            )
        if tdobj.deltaS == 1:
            return delta_s_plus_one.grad_elec(
                self, tdobj, xy, **options,
            )
        raise ValueError("deltaS must be -1, 0, or 1")

    def grad_elec(self, xy, atmlst=None):
        """Return the analytic excitation-energy derivative ``d omega/dR``."""
        if atmlst is None:
            atmlst = range(self.mol.natm)
        components = self._analytic_components(xy, tuple(atmlst))
        self.nttda_details = components
        return components.total

    def _energy_at(self, coords, reference_amplitude):
        mol = self.mol.copy()
        mol.set_geom_(coords, unit="Bohr")
        mf = _displaced_reference(self.base._scf, mol, self.fixed_grid)
        mf.kernel(dm0=self.base._scf.make_rdm1())
        if not mf.converged:
            raise RuntimeError("displaced ROKS reference did not converge")
        tdobj = _copy_td_settings(self.base, NTTDA(mf))
        tdobj.kernel()
        overlaps = np.asarray([
            abs(np.vdot(reference_amplitude, _normalized_amplitude(xy)))
            for xy in tdobj.xy
        ])
        root = int(np.argmax(overlaps))
        if overlaps[root] < self.root_overlap_tol:
            raise RuntimeError(
                "NTTDA state tracking overlap %.6f is below %.6f" %
                (overlaps[root], self.root_overlap_tol)
            )
        return mf.e_tot + tdobj.e[root]

    def _finite_difference(self, atmlst):
        coords0 = self.mol.atom_coords()
        reference_amplitude = _normalized_amplitude(
            self.base.xy[self.state - 1],
        )
        result = np.zeros((len(atmlst), 3))
        for index, atom in enumerate(atmlst):
            for xyz in range(3):
                coords_plus = coords0.copy()
                coords_minus = coords0.copy()
                coords_plus[atom, xyz] += self.step
                coords_minus[atom, xyz] -= self.step
                energy_plus = self._energy_at(
                    coords_plus, reference_amplitude,
                )
                energy_minus = self._energy_at(
                    coords_minus, reference_amplitude,
                )
                result[index, xyz] = (
                    (energy_plus - energy_minus) / (2.0 * self.step)
                )
        return result

    def kernel(self, state=None, atmlst=None, method=None, step=None):
        """Return ``d(E_ROKS + omega_state)/dR`` in Eh/Bohr."""
        if state is not None:
            self.state = state
        if method is not None:
            self.method = method
        if step is not None:
            self.step = step
        if atmlst is None:
            atmlst = self.atmlst
        else:
            self.atmlst = atmlst
        if atmlst is None:
            atmlst = range(self.mol.natm)
        atmlst = tuple(atmlst)

        if self.state == 0:
            return self.grad_nuc(atmlst=atmlst)
        if self.base.xy is None:
            self.base.run()
        if not 1 <= self.state <= len(self.base.xy):
            raise ValueError("state must be in [1, %d]" % len(self.base.xy))
        if self.verbose >= logger.INFO:
            self.dump_flags()

        if self.method == "analytic":
            excitation = self.grad_elec(
                self.base.xy[self.state - 1], atmlst=atmlst,
            )
            result = self.grad_nuc(atmlst=atmlst) + excitation
        elif self.method == "finite_diff":
            result = self._finite_difference(atmlst)
        else:
            raise ValueError("unknown NTTDA gradient method %s" % self.method)
        self.de = result
        if self.mol.symmetry:
            self.de = self.symmetrize(self.de, atmlst)
        self._finalize()
        return self.de

    grad = lib.alias(kernel, alias_name="grad")


Grad = Gradients

__all__ = ["Grad", "Gradients"]
