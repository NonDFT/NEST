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

"""Common machinery for spin-orbit-coupled excited states."""

from dataclasses import dataclass
from math import sqrt
from typing import Any

import numpy as np
from pyscf import lib
from pyscf.data.nist import HARTREE2EV, HARTREE2WAVENUMBER
from pyscf.lib import logger

from nest.soc.soc_ao import get_ao_soc


@dataclass
class SpinFreeState:
    """One scalar excited state before spin-orbit coupling is introduced."""

    source: Any
    root: int | None
    energy: float
    spin: float
    amplitude: Any
    label: str
    spin_square: float | None = None
    delta_s: int | None = None


def clebsch_gordan_rank1(j1, m1, q, j, m):
    """Return ``<j1,m1;1,q|j,m>`` for the rank-one SOC tensor."""
    tol = 1e-12
    if q not in (-1, 0, 1) or abs(m - (m1 + q)) > tol:
        return 0.0
    if j1 < 0 or j < 0 or abs(m1) > j1 + tol or abs(m) > j + tol:
        return 0.0

    if abs(j - (j1 + 1)) < tol:
        if q == 1:
            return sqrt((j1 + m1 + 1) * (j1 + m1 + 2) / (2 * (j1 + 1) * (2 * j1 + 1)))
        if q == 0:
            return sqrt((j1 - m1 + 1) * (j1 + m1 + 1) / ((j1 + 1) * (2 * j1 + 1)))
        return sqrt((j1 - m1 + 1) * (j1 - m1 + 2) / (2 * (j1 + 1) * (2 * j1 + 1)))

    if abs(j - j1) < tol:
        if j1 == 0:
            return 0.0
        if q == 1:
            return -sqrt((j1 - m1) * (j1 + m1 + 1) / (2 * j1 * (j1 + 1)))
        if q == 0:
            return m1 / sqrt(j1 * (j1 + 1))
        return sqrt((j1 + m1) * (j1 - m1 + 1) / (2 * j1 * (j1 + 1)))

    if abs(j - (j1 - 1)) < tol:
        if j1 == 0:
            return 0.0
        if q == 1:
            return sqrt((j1 - m1) * (j1 - m1 - 1) / (2 * j1 * (2 * j1 + 1)))
        if q == 0:
            return -sqrt((j1 - m1) * (j1 + m1) / (j1 * (2 * j1 + 1)))
        return sqrt((j1 + m1) * (j1 + m1 - 1) / (2 * j1 * (2 * j1 + 1)))
    return 0.0


class SOCBase(lib.StreamObject):
    """Build and diagonalize a SOC Hamiltonian from scalar excited states."""

    _keys = {'states', 'soctype', 'soc_ao', 'state_slices', 'h_soc', 'e', 'v'}

    def __init__(self, mf, states, soctype='SOMF'):
        self._scf = mf
        self.states = list(states)
        self.soctype = soctype
        self.verbose = getattr(mf, 'verbose', logger.NOTE)
        self.stdout = getattr(mf, 'stdout', None)

        self.soc_ao = None
        self.state_slices = None
        self.h_soc = None
        self.e = None
        self.v = None

    @staticmethod
    def _m_values(spin):
        values = np.arange(-spin, spin + 0.5, 1.0)
        values[abs(values) < 1e-12] = 0.0
        return values

    def reduced_transition_density(self, bra, ket):
        r"""Return the AO reduced transition density for ``S_bra >= S_ket``.
        \gamma_{\nu\mu} = <bra||T_{\mu\nu}||ket>
        """
        raise NotImplementedError

    def _soc_block(self, bra, ket):
        nbra = int(round(2 * bra.spin + 1))
        nket = int(round(2 * ket.spin + 1))
        if abs(bra.spin - ket.spin) > 1 + 1e-12:
            return np.zeros((nbra, nket), dtype=np.complex128)
        if bra.spin + 1e-12 < ket.spin:
            return self._soc_block(ket, bra).conj().T

        density = self.reduced_transition_density(bra, ket)
        # <bra|z|ket> = sum_uv z_uv gamma_vu = Tr(z gamma).
        # ``density[v, u]`` therefore contracts with ``soc_ao[x, u, v]``.
        components = np.einsum('xuv,vu->x', self.soc_ao, density)
        block = np.zeros((nbra, nket), dtype=np.complex128)
        for row, m_bra in enumerate(self._m_values(bra.spin)):
            for col, m_ket in enumerate(self._m_values(ket.spin)):
                q = int(round(m_bra - m_ket))
                if abs(m_bra - m_ket - q) > 1e-12 or q not in (-1, 0, 1):
                    continue
                coefficient = clebsch_gordan_rank1(ket.spin, m_ket, q, bra.spin, m_bra)
                if q == 1:
                    block[row, col] = -coefficient * components[0]
                elif q == 0:
                    block[row, col] = coefficient * components[1]
                else:
                    block[row, col] = -coefficient * components[2]
        return block

    def build_hamiltonian(self):
        self.soc_ao = get_ao_soc(self._scf, self.soctype)
        self.state_slices = []
        start = 0
        for state in self.states:
            stop = start + len(self._m_values(state.spin))
            self.state_slices.append(slice(start, stop))
            start = stop
        h_soc = np.zeros((start, start), dtype=np.complex128)

        # Diagonal first-order SOC blocks vanish for the real scalar states
        # accepted by the method-specific drivers.
        for state, state_slice in zip(self.states, self.state_slices):
            dimension = state_slice.stop - state_slice.start
            h_soc[state_slice, state_slice] = np.eye(dimension) * state.energy

        for bra_id in range(len(self.states)):
            for ket_id in range(bra_id):
                block = self._soc_block(self.states[bra_id], self.states[ket_id])
                bra_slice = self.state_slices[bra_id]
                ket_slice = self.state_slices[ket_id]
                h_soc[bra_slice, ket_slice] = block
                h_soc[ket_slice, bra_slice] = block.conj().T

        error = np.max(abs(h_soc - h_soc.conj().T)) if h_soc.size else 0.0
        if error > 1e-10:
            raise ValueError(f'SOC Hamiltonian is not Hermitian: max error {error:.3e}')
        self.h_soc = h_soc
        return h_soc

    def kernel(self):
        self.build_hamiltonian()
        self.e, self.v = np.linalg.eigh(self.h_soc)
        return self.e, self.v

    def get_block(self, bra, ket):
        if self.h_soc is None:
            raise RuntimeError('Run kernel() or build_hamiltonian() first')
        return self.h_soc[self.state_slices[bra], self.state_slices[ket]]

    @classmethod
    def _format_soc_block(cls, block, bra, ket):
        """Format one SOC block with explicit bra/ket ``M_S`` labels."""
        bra_m_values = cls._m_values(bra.spin)
        ket_m_values = cls._m_values(ket.spin)
        expected_shape = (len(bra_m_values), len(ket_m_values))
        if block.shape != expected_shape:
            raise ValueError(
                f'SOC block shape {block.shape} does not match the expected '
                f'bra/ket dimensions {expected_shape}'
            )

        label_width = 18
        column_width = 27
        header = ' ' * label_width
        for m_ket in ket_m_values:
            header += f'ket M_S={m_ket:5.1f}'.center(column_width)

        lines = ['SOC matrix elements (cm^-1):', header, '-' * len(header)]
        for row, m_bra in enumerate(bra_m_values):
            line = f'bra M_S={m_bra:5.1f}'.ljust(label_width)
            for value in block[row]:
                line += f'({value.real:10.6f},{value.imag:10.6f})'.center(column_width)
            lines.append(line)
        return lines

    def analyze(self, verbose=None):
        if self.h_soc is None or self.e is None:
            self.kernel()
        log = logger.new_logger(self, verbose)
        log.note('SOC scalar states')
        for state_id, state in enumerate(self.states):
            message = 'State %d: %s  S=%s  E=%.8f Eh' % (
                state_id + 1, state.label, state.spin, state.energy,
            )
            if state.spin_square is not None:
                message += '  <S^2>=%.6f' % state.spin_square
            log.note(message)

        for bra_id in range(len(self.states)):
            for ket_id in range(bra_id):
                block = self.get_block(bra_id, ket_id)
                bra = self.states[bra_id]
                ket = self.states[ket_id]
                log.note(
                    'SOC block: state %d (%s, S=%s) <- state %d (%s, S=%s); '
                    'SOCC = %.6f cm^-1',
                    bra_id + 1, bra.label, bra.spin,
                    ket_id + 1, ket.label, ket.spin,
                    np.linalg.norm(block) * HARTREE2WAVENUMBER,
                )
                for line in self._format_soc_block(
                    block * HARTREE2WAVENUMBER, bra, ket,
                ):
                    log.info('%s', line)

        origin = self.e.min() if self.e.size else 0.0
        log.note('Spin-orbit-coupled eigenstates')
        for state_id, energy in enumerate(self.e):
            log.note('State %d: Delta E = %.6f cm^-1 (%.8f eV)', state_id + 1,
                     (energy - origin).real * HARTREE2WAVENUMBER,
                     (energy - origin).real * HARTREE2EV)
            weights = abs(self.v[:, state_id]) ** 2
            log.info('  Spin-free state composition:')
            for state, state_slice in zip(self.states, self.state_slices):
                probability = weights[state_slice].sum()
                if probability > 0.01:
                    log.info('    %5.1f%% from %s (S=%s)',
                             probability * 100, state.label, state.spin)
        return self


__all__ = ['SpinFreeState', 'SOCBase', 'clebsch_gordan_rank1']
