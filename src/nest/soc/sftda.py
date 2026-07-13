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

"""SOC driver for spin-flip TDA and TDDFT."""

import numpy as np
from pyscf import lib
from pyscf.lib import logger

from nest.soc.soc import SOCBase, SpinFreeState, clebsch_gordan_rank1


class SOC(SOCBase):
    """Build SOC states from one converged SFTDA or SFTDDFT object."""

    _keys = {'tdobj', 'sz'}

    def __init__(self, tdobj, soctype='SOMF'):
        if tdobj.extype != 1:
            raise ValueError('SFTDA SOC currently supports extype=1 only')
        if getattr(tdobj, 'e', None) is None or getattr(tdobj, 'xy', None) is None:
            raise ValueError('Run the SFTDA/SFTDDFT kernel before SOC')
        assert np.isrealobj(tdobj._scf.mo_coeff), 'SFTDA SOC requires real MO coefficients'

        spin_square = np.asarray(tdobj.spin_square(), dtype=float)
        states = []
        for root, energy in enumerate(tdobj.e):
            s2 = float(spin_square[root])
            spin = round(-1 + np.sqrt(1 + 4 * s2)) / 2
            amplitude = tdobj.xy[root]
            assert np.isrealobj(amplitude[0]), 'SFTDA SOC requires real X amplitudes'
            if isinstance(amplitude[1], np.ndarray):
                assert np.isrealobj(amplitude[1]), 'SFTDDFT SOC requires real Y amplitudes'
            states.append(SpinFreeState(
                source=tdobj,
                root=root,
                energy=float(energy),
                spin=spin,
                amplitude=amplitude,
                label=f'SF state {root + 1}',
                spin_square=s2,
            ))
        super().__init__(tdobj._scf, states, soctype=soctype)
        self.tdobj = tdobj
        self.sz = 0.5 * self._scf.mol.spin - 1

    def reduced_transition_density(self, bra, ket):
        nao = self._scf.mol.nao_nr()
        if abs(bra.spin) < 1e-12 and abs(ket.spin) < 1e-12:
            # A rank-one operator cannot couple two singlets.
            return np.zeros((nao, nao), dtype=np.complex128)

        # Extract the reduced matrix element from the q=0 component at the
        # spin projection represented by the spin-flip amplitudes.  A zero
        # coefficient here does not generally imply a zero reduced matrix
        # element; it means that this q=0 component contains no information
        # about it (for example, S=1, M_S=0 <-> S=1, M_S=0).
        coefficient = clebsch_gordan_rank1(ket.spin, self.sz, 0, bra.spin, self.sz)
        if abs(coefficient) < 1e-12:
            logger.warn(
                self,
                'The q=0 SF transition density cannot determine the reduced '
                'SOC matrix element for S=%s <- S=%s at M_S=%s. Returning zero; '
                'the affected states may be spin contaminated.',
                bra.spin, ket.spin, self.sz,
            )
            return np.zeros((nao, nao), dtype=np.complex128)

        mo_coeff = self._scf.mo_coeff
        mo_occ = self._scf.mo_occ
        occidxa = mo_occ[0] > 0
        occidxb = mo_occ[1] > 0
        viridxa = mo_occ[0] == 0
        viridxb = mo_occ[1] == 0
        orboa = mo_coeff[0][:, occidxa]
        orbob = mo_coeff[1][:, occidxb]
        orbva = mo_coeff[0][:, viridxa]
        orbvb = mo_coeff[1][:, viridxb]

        mx = bra.amplitude[0]
        nx = ket.amplitude[0]
        gamma_oo_aa = -lib.einsum('ia,ja->ij', mx, nx)
        gamma_aa = lib.einsum('uj,vi,ij->vu', orboa, orboa, gamma_oo_aa)
        gamma_vv_bb = lib.einsum('ib,ia->ab', mx, nx)
        gamma_bb = lib.einsum('ub,va,ab->vu', orbvb, orbvb, gamma_vv_bb)

        if isinstance(bra.amplitude[1], np.ndarray):
            my = bra.amplitude[1]
            ny = ket.amplitude[1]
            gamma_oo_bb = -lib.einsum('ja,ia->ij', my, ny)
            gamma_bb += lib.einsum('uj,vi,ij->vu', orbob, orbob, gamma_oo_bb)
            gamma_vv_aa = lib.einsum('ia,ib->ab', my, ny)
            gamma_aa += lib.einsum('ub,va,ab->vu', orbva, orbva, gamma_vv_aa)

        # The AO transformations above explicitly output ``gamma[v, u]``.
        # Keep that physical transition-density order; SOCBase contracts it
        # as z[u, v] * gamma[v, u].
        return (gamma_aa - gamma_bb) / np.sqrt(2) / coefficient


__all__ = ['SOC']
