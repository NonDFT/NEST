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

"""Analytic excitation gradient for the NTTDA ``deltaS=+1`` channel.

Only closed-to-virtual amplitudes occur. The excitation scalar is
``(Dp - Dh):F0 + (Dp + Dh):Fz + T:K0[T]``: the hole has opposite F0
and Fz signs, and the response has no vref1 term. See
``docs/derivations/nttda_delta_s_plus_one_gradient.md`` for the derivation.
"""

import numpy as np

from nest.nttda.nttda import gen_rohf_response_sfu, _split_amplitudes

from .common import (
    FockProjection,
    ResponseTerm,
    _apply_reference_responses,
    fock_probes,
    orbital_spaces,
    pair_density,
)
from .driver import assemble_gradient


RESPONSE_TERMS = (ResponseTerm('CV', 'CV', 1.0, 0.0),)


def split_spin_raising(tdobj, xy):
    """Return orbital spaces and the native ``X[i,a]`` amplitude matrix."""
    spaces = orbital_spaces(tdobj)
    vector = xy[0] if isinstance(xy, (tuple, list)) else xy
    vector = np.asarray(vector).reshape(-1)
    nc, no, nv = len(spaces.closed), len(spaces.open), len(spaces.virtual)
    if vector.size != nc * nv:
        raise ValueError('deltaS=+1 amplitude has size %d; expected %d' % (vector.size, nc * nv))
    _, cv, _, _, _ = _split_amplitudes(vector, 1, nc, no, nv)
    return spaces, cv


def fock0_fockz(tdobj, max_memory=None):
    """Build the same F0 and Fz as the production spin-raising action."""
    mf = tdobj._scf
    if max_memory is None:
        max_memory = tdobj.max_memory
    _response, fockz = gen_rohf_response_sfu(
        mf,
        mo_coeff=mf.mo_coeff,
        mo_occ=mf.mo_occ,
        hermi=0,
        max_memory=max_memory,
    )
    if tdobj.nobeta:
        dma, dmb = mf.make_rdm1()
        dm0 = 0.5 * (dma + dmb)
        fock = mf.get_fock(dm=np.array([dm0, dm0]))
    else:
        fock = mf.get_fock()
    return 0.5 * (fock.focka + fock.fockb), fockz


def spin_raising_fock_projections(tdobj, xy):
    """Particle minus hole for F0; particle plus hole for Fz."""
    spaces, cv = split_spin_raising(tdobj, xy)
    c, v = spaces.closed, spaces.virtual
    return (
        FockProjection('cv-vv', v, cv.T @ cv, v, 1.0, 1.0),
        FockProjection('cv-cc', c, cv @ cv.T, c, -1.0, 1.0),
    )


def spin_raising_ledger_scalar(tdobj, xy, max_memory=None, return_parts=False):
    """Evaluate the derived scalar independently of ``gen_vind_sfu``."""
    spaces, cv = split_spin_raising(tdobj, xy)
    density = pair_density(spaces.c_virtual, cv.T, spaces.c_closed)
    p0, pz = fock_probes(tdobj, spin_raising_fock_projections(tdobj, xy))
    fock0, fockz = fock0_fockz(tdobj, max_memory=max_memory)
    fock = float(np.sum(p0 * fock0) + np.sum(pz * fockz))
    vref0, _vref1 = _apply_reference_responses(
        tdobj,
        {'CV': density},
        max_memory=max_memory,
    )
    response = float(np.sum(density * vref0['CV']))
    total = fock + response
    if return_parts:
        return {'fock': fock, 'response': response, 'total': total}
    return total


def grad_elec(gradient_driver, tdobj, xy, atmlst=None, tolerance=1e-12, max_cycle=None):
    """Build d(omega)/dR from two Fock projections and one CV response."""
    if tdobj.deltaS != 1:
        raise ValueError('deltaS=+1 gradient received a different spin channel')
    spaces, cv = split_spin_raising(tdobj, xy)
    return assemble_gradient(
        gradient_driver,
        tdobj,
        spaces=spaces,
        densities={'CV': pair_density(spaces.c_virtual, cv.T, spaces.c_closed)},
        blocks={'CV': (spaces.virtual, spaces.closed, cv.T)},
        response_terms=RESPONSE_TERMS,
        projections=spin_raising_fock_projections(tdobj, xy),
        operators=fock0_fockz(tdobj, max_memory=gradient_driver.max_memory),
        atmlst=atmlst,
        tolerance=tolerance,
        max_cycle=max_cycle,
    )
