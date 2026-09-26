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

"""Analytic nuclear gradients for :mod:`nest.nttda`."""

import numpy as np

from pyscf import dft, gto, lib
from pyscf.grad import rhf as rhf_grad
from pyscf.grad import tdrhf as tdrhf_grad
from pyscf.lib import logger
from nest.nttda import NTTDA
from nest.nttda.nttda import _is_average_occupation_reference

from . import delta_s_minus_one, delta_s_zero



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
    if _is_average_occupation_reference(source):
        reference = dft.ROKS(mol).average_occ()
    elif isinstance(source, dft.KohnShamDFT):
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
        if fixed_grid and source.grids.coords is not None:
            reference.grids.coords = np.array(source.grids.coords, copy=True)
            reference.grids.weights = np.array(source.grids.weights, copy=True)
            reference.grids.non0tab = None
            reference.grids.verbose = 0
    return reference


class TDSCF_GradScanner(tdrhf_grad.TDSCF_GradScanner):
    def __call__(self, mol_or_geom, state=None, **kwargs):
        if state is not None:
            self.state = state
        if self.state != 0:
            return super().__call__(mol_or_geom, state=self.state, **kwargs)
        if isinstance(mol_or_geom, gto.MoleBase):
            mol = mol_or_geom
        else:
            mol = self.mol.set_geom_(mol_or_geom, inplace=False)
        self.reset(mol)
        mf = self.base._scf
        mf(mol)
        return mf.e_tot, self.kernel(state=0, **kwargs)

    @property
    def converged(self):
        mf_converged = self.base._scf.converged
        if self.state == 0:
            return mf_converged
        return mf_converged and self.base.converged[self.state - 1]


class Gradients(rhf_grad.GradientsBase):
    """NTTDA analytic and finite-difference gradients for ROKS and AOCSCF.

    Analytic DFT derivatives omit grid response. ``fixed_grid`` controls only
    the finite-difference grids; keep it true when checking analytic gradients.
    """

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

    def as_scanner(self, state=None):
        """Return the selected state's total energy and nuclear gradient."""
        if isinstance(self, lib.GradScanner):
            if state is not None:
                self.state = state
            return self
        name = self.__class__.__name__ + TDSCF_GradScanner.__name_mixin__
        return lib.set_class(TDSCF_GradScanner(self, state),
                             (TDSCF_GradScanner, self.__class__), name)

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
        """Ground-state reference gradient, including nuclear repulsion."""
        if atmlst is not None:
            atmlst = list(atmlst)
        reference_gradient = self.base._scf.nuc_grad_method()
        if self.mol.symmetry:
            de = reference_gradient.kernel()
            return de if atmlst is None else de[atmlst]
        return reference_gradient.kernel(atmlst=atmlst)

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
            raise NotImplementedError(
                "Analytic NTTDA gradients are not implemented for deltaS=1; "
                "use method='finite_diff'."
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
            raise RuntimeError("displaced NTTDA reference did not converge")
        tdobj = _copy_td_settings(self.base, NTTDA(mf))
        tdobj.kernel()
        overlaps = np.asarray([
            abs(np.vdot(reference_amplitude, _normalized_amplitude(xy)))
            for xy in tdobj.xy
        ])
        root = int(np.argmax(overlaps))
        if not tdobj.converged[root]:
            raise RuntimeError("displaced NTTDA state did not converge")
        if overlaps[root] < self.root_overlap_tol:
            raise RuntimeError(
                "NTTDA state tracking overlap %.6f is below %.6f" %
                (overlaps[root], self.root_overlap_tol)
            )
        return float(tdobj.total_energies()[root])

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
        """Return ``d(E_reference + omega_state)/dR`` in Eh/Bohr."""
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
            self.de = self.grad_nuc(atmlst=atmlst)
            return self.de
        if self.base.xy is None:
            self.base.run()
        if not 1 <= self.state <= len(self.base.xy):
            raise ValueError("state must be in [1, %d]" % len(self.base.xy))
        if not self.base._scf.converged or not self.base.converged[self.state - 1]:
            raise RuntimeError("converge the SCF reference and selected NTTDA state before computing a gradient")
        # The derivative integrals below use the conventional molecular AO Hamiltonian.
        mf = self.base._scf
        for attribute, label in (("with_df", "Density-fitted"),
                                 ("with_x2c", "X2C"),
                                 ("with_solvent", "Solvent-response")):
            if getattr(mf, attribute, None) is not None:
                raise NotImplementedError("%s NTTDA gradients are not implemented" % label)
        if mf.do_nlc():
            raise NotImplementedError("NLC NTTDA gradients are not implemented")
        if self.verbose >= logger.INFO:
            self.dump_flags()

        # Symmetrize the full gradient before selecting atoms.
        calculation_atoms = tuple(range(self.mol.natm)) if self.mol.symmetry else atmlst
        if self.method == "analytic":
            excitation = self.grad_elec(
                self.base.xy[self.state - 1], atmlst=calculation_atoms,
            )
            result = self.grad_nuc(atmlst=calculation_atoms) + excitation
        elif self.method == "finite_diff":
            result = self._finite_difference(calculation_atoms)
        else:
            raise ValueError("unknown NTTDA gradient method %s" % self.method)
        self.de = result
        if self.mol.symmetry:
            self.de = self.symmetrize(self.de)[list(atmlst)]
        self._finalize()
        return self.de

    grad = lib.alias(kernel, alias_name="grad")


Grad = Gradients

__all__ = ["Grad", "Gradients"]
