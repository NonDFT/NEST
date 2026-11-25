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
and Fz signs, and the response has no vref1 term.
"""

import numpy as np

from nest.nttda.nttda import gen_rohf_response_sfu
from .common import (
    FockProjection, ResponseTerm, assemble_gradient, fock_probes,
    fock_projection_q, orbital_spaces, pair_density, response_projection_q,
)


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


def grad_elec(gradient_driver, tdobj, xy, atmlst=None, tolerance=1e-12, max_cycle=None):
    """Differentiate the closed-to-virtual spin-raising excitation."""
    spaces = orbital_spaces(tdobj)
    c, v = spaces.closed, spaces.virtual
    cc, cv = spaces.c_closed, spaces.c_virtual
    vector = xy[0] if isinstance(xy, (tuple, list)) else xy
    x = np.asarray(vector).reshape(len(c), len(v))
    projections = (
        FockProjection('particle', v, cv, x.T @ x, v, cv, 1., 1.),
        FockProjection('hole', c, cc, x @ x.T, c, cc, -1., 1.),
    )
    densities = {'CV': pair_density(cv, x.T, cc)}
    blocks = {'CV': (v, c, x.T)}
    terms = (ResponseTerm('CV', 'CV', 1., 0.),)
    channel = (spaces, x, densities, blocks, terms)
    probes = fock_probes(tdobj, projections)
    operators = fock0_fockz(tdobj, gradient_driver.max_memory)
    return assemble_gradient(
        gradient_driver, tdobj, channel, probes,
        fock_projection_q(tdobj, projections, operators, probes),
        response_projection_q(tdobj, channel, hfx_only=True),
        atmlst=atmlst, tolerance=tolerance, max_cycle=max_cycle,
    )
