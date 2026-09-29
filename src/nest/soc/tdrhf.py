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

"""
SOC driver framework for closed-shell RHF/RKS TDA and TDDFT.
Ref: J. Chem. Theory Comput. 2019, 15, 1896.
"""

import numpy as np
from pyscf import lib
from pyscf.lib import logger
from pyscf.tdscf import dhf, ghf, rhf, uhf

from nest.soc.soc import SOCBase, SpinFreeState


class SOC(SOCBase):
    """Build scalar states from closed-shell singlet and triplet TD objects.

    A state from a TD object with ``singlet=True`` has ``S=0``; one with
    ``singlet=False`` has ``S=1``.
    """

    _keys = {'tds', 'include_reference'}

    def __init__(self, td0=None, *others, soctype='SOMF', include_reference=True):
        if td0 is None:
            if others:
                raise ValueError('The first RHF/RKS TD object cannot be None')
            tds = []
        elif isinstance(td0, (list, tuple)) and not others:
            tds = list(td0)
        else:
            tds = [td0, *others]

        super().__init__(soctype=soctype)
        self.tds = tds
        self.include_reference = include_reference

    def _initialize_states(self):
        if not isinstance(self.tds, (list, tuple)):
            raise TypeError('tds must be a list or tuple of RHF/RKS TD objects')
        if not self.tds:
            raise ValueError('Set tds to at least one RHF/RKS TD object before running SOC')

        mf = self._check_td_objects(self.tds)
        self._scf = mf
        if self.verbose is None:
            self.verbose = getattr(mf, 'verbose', logger.NOTE)
        if self.stdout is None:
            self.stdout = getattr(mf, 'stdout', None)

        log = logger.new_logger(self)
        log.info('\n')
        if len(self.tds) == 1:
            log.warn('Only one RHF/RKS TD object is included in the SOC calculation')
        if not getattr(mf, 'converged', False):
            log.warn('Ground state SCF is not converged')

        states = []
        if self.include_reference:
            states.append(SpinFreeState(
                source=None,
                root=None,
                energy=0.0,
                spin=0.0,
                amplitude=None,
                label='reference',
            ))

        for tdobj in self.tds:
            if getattr(tdobj, 'e', None) is None or getattr(tdobj, 'xy', None) is None:
                raise ValueError('Run every RHF/RKS TD kernel before SOC')

            converged = getattr(tdobj, 'converged', None)
            if converged is not None:
                unconverged = np.where(~np.asarray(converged, dtype=bool).reshape(-1))[0]
                if unconverged.size:
                    log.warn('TD states %s are not converged', unconverged.tolist())

            spin = 0.0 if tdobj.singlet else 1.0
            multiplicity = 'singlet' if tdobj.singlet else 'triplet'
            for root, energy in enumerate(tdobj.e):
                amplitude = self._state_amplitude(tdobj, root)
                states.append(SpinFreeState(
                    source=tdobj,
                    root=root,
                    energy=float(energy),
                    spin=spin,
                    amplitude=amplitude,
                    label=f'{multiplicity} state {root + 1}',
                ))
        return states

    @staticmethod
    def _state_amplitude(tdobj, root):
        """Return the normalized Casida pseudo-wavefunction amplitude."""
        x, y = tdobj.xy[root]
        if not np.isrealobj(x) or not np.isrealobj(y):
            raise ValueError('Closed-shell TD SOC requires real X/Y amplitudes')

        x = np.asarray(x)
        if isinstance(y, np.ndarray):
            amplitude = x + np.asarray(y)
            description = 'X+Y'
        elif np.isscalar(y) and y == 0:
            amplitude = x
            description = 'X'
        else:
            raise ValueError('TD amplitudes must be stored as (X, Y) for TDDFT or (X, 0) for TDA')

        norm = np.linalg.norm(amplitude)
        if norm == 0:
            raise ValueError(f'TD state {root + 1} has a zero-norm {description} amplitude')
        return amplitude / norm

    @staticmethod
    def _check_td_objects(tdobjs):
        unsupported = (uhf.TDBase, ghf.TDBase, dhf.TDBase)
        for tdobj in tdobjs:
            if type(getattr(tdobj, 'singlet', None)) is not bool:
                raise ValueError('Each RHF/RKS TD object must set singlet to True or False')
            if isinstance(tdobj, unsupported) or not isinstance(tdobj, rhf.TDBase):
                raise TypeError('Closed-shell TD SOC accepts only RHF/RKS TDA or TDDFT objects')

        mf = tdobjs[0]._scf
        mo_occ = np.asarray(mf.mo_occ)
        mo_coeff = np.asarray(mf.mo_coeff)
        if mo_occ.ndim != 1 or np.any((mo_occ != 0) & (mo_occ != 2)):
            raise ValueError('Closed-shell TD SOC requires occupations containing only 0 or 2')
        if mo_coeff.ndim != 2 or not np.isrealobj(mo_coeff):
            raise ValueError('Closed-shell TD SOC requires real, spin-restricted MO coefficients')

        for tdobj in tdobjs[1:]:
            other_mf = tdobj._scf
            if not np.array_equal(np.asarray(other_mf.mo_occ), mo_occ):
                raise ValueError('Both TD objects must share the same mo_occ')
            other_coeff = np.asarray(other_mf.mo_coeff)
            if not np.isrealobj(other_coeff):
                raise ValueError('Closed-shell TD SOC requires real, spin-restricted MO coefficients')
            if not np.allclose(other_coeff, mo_coeff, atol=1e-12, rtol=0):
                raise ValueError('Both TD objects must share the same mo_coeff')
        return mf

    def reduced_transition_density(self, bra, ket):
        """Return ``<bra||T_pq||ket>`` in AO ``[q, p]`` order."""
        if bra.spin + 1e-12 < ket.spin:
            raise ValueError('Reduced transition density requires bra.spin >= ket.spin')

        mo_occ = np.asarray(self._scf.mo_occ)
        occidx = np.where(mo_occ == 2)[0]
        viridx = np.where(mo_occ == 0)[0]
        nmo = mo_occ.size
        gamma_mo = np.zeros((nmo, nmo))

        if abs(bra.spin) < 1e-12:
            pass
        elif abs(ket.spin) < 1e-12 and ket.source is None:
            gamma_mo[np.ix_(viridx, occidx)] = bra.amplitude.T
        elif abs(ket.spin) < 1e-12:
            triplet = bra.amplitude
            singlet = ket.amplitude
            factor = 1.0 / np.sqrt(2.0)
            gamma_mo[np.ix_(viridx, viridx)] = factor * triplet.T @ singlet
            gamma_mo[np.ix_(occidx, occidx)] = -factor * singlet @ triplet.T
        else:
            bra_triplet = bra.amplitude
            ket_triplet = ket.amplitude
            gamma_mo[np.ix_(viridx, viridx)] = bra_triplet.T @ ket_triplet
            gamma_mo[np.ix_(occidx, occidx)] = ket_triplet @ bra_triplet.T

        coeff = np.asarray(self._scf.mo_coeff)
        # gamma_mo[p,q] is transformed to the density[q,p] convention used by
        # SOCBase, which contracts z[p,q] * density[q,p].
        return lib.einsum('up,pq,vq->uv', coeff.conj(), gamma_mo, coeff).T


__all__ = ['SOC']
