#!/usr/bin/env python
# Copyright 2014-2024 The PySCF Developers. All Rights Reserved.
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

"""Internal iterative eigensolvers shared by NEST response methods."""

import sys
import numpy as np
import scipy
from pyscf.lib import logger
from pyscf.lib.parameters import MAX_MEMORY
from pyscf.lib.linalg_helper import _sort_elast
from pyscf.tdscf._lr_eig import MAX_SPACE_INC


def davidson_nosym1(aop, x0, precond, tol=1e-12, max_cycle=50, lindep=1e-12, callback=None,
                    max_memory=MAX_MEMORY, nroots=1, pick=None, verbose=logger.WARN):
    '''
    slightly modified from pyscf.lib.linalg.davidson_nosym1 with vectorization support
    '''

    assert callable(pick)
    assert callable(precond)

    if isinstance(verbose, logger.Logger):
        log = verbose
    else:
        log = logger.Logger(sys.stdout, verbose)

    toloose = tol ** 0.5
    log.debug1('tol %g  toloose %g', tol, toloose)

    if isinstance(x0, np.ndarray) and x0.ndim == 1:
        x0 = x0[None, :]
    x0 = np.asarray(x0)
    x0_size = x0.shape[1]

    if MAX_SPACE_INC is None:
        space_inc = nroots
    else:
        space_inc = min(nroots, min(MAX_SPACE_INC, x0_size//2))
    max_space = int(max_memory*1e6/8/x0_size / 2 - nroots - space_inc)
    if max_space < nroots * 4 < x0_size:
        log.warn('Not enough memory to store trial space in _lr_eig.eigh')
    max_space = max(max_space, nroots * 4)
    max_space = min(max_space, x0_size)
    log.debug(f'Set max_space {max_space}, space_inc {space_inc}')

    xs = np.empty((max_space, x0_size), dtype=x0.dtype)
    ax = np.empty((max_space, x0_size), dtype=x0.dtype)
    heff = np.empty((max_space, max_space), dtype=x0.dtype)
    fresh_start = True
    space = 0
    e = None
    v = None
    conv = np.zeros(nroots, dtype=bool)
    conv_last = np.zeros(nroots, dtype=bool)

    for icyc in range(max_cycle):
        if fresh_start:
            xs = np.empty_like(xs)
            ax = np.empty_like(ax)
            space = 0
            x0len = len(x0)
            xt, _ = _qr(x0, lindep)
            if len(xt) != x0len:
                log.warn('QR decomposition removed %d vectors. '
                            'Check to see if `pick` function :%s: is providing linear dependent '
                            'vectors' % (x0len - len(xt), pick.__name__))
        elif len(xt) > 1:
            xt, _ = _qr(xt, lindep)
        if icyc != 0:
            xt = xt[:space_inc]

        add = xt.shape[0]
        axt = aop(xt)
        xs[space: space+add] = xt
        ax[space: space+add] = axt
        space_old = space
        space += add
        if fresh_start:
            heff.fill(0)
        heff[:space_old, space_old:space] = xs[:space_old].conj().dot(ax[space_old:space].T)
        heff[space_old:space, :space_old] = xs[space_old:space].conj().dot(ax[:space_old].T)
        heff[space_old:space, space_old:space] = xs[space_old:space].conj().dot(ax[space_old:space].T)
        xt = axt = None

        elast = e
        vlast = v
        conv_last = conv

        w_npu, v_npu = scipy.linalg.eig(heff[:space,:space])
        w, v = np.asarray(w_npu), np.asarray(v_npu)
        w, v, idx = pick(w, v, nroots, locals())
        if len(w) == 0:
            raise RuntimeError('Not enough eigenvalues')

        e = w[:nroots]
        v = v[:,:nroots]
        if not fresh_start:
            elast, conv_last = _sort_elast(elast, conv_last, vlast, v, log)

        if elast is None:
            de = e
        elif elast.size != e.size:
            log.debug('Number of roots different from the previous step (%d,%d)',
                      e.size, elast.size)
            de = e
        else:
            de = e - elast

        x0 = v.T.dot(xs[:space])
        ax0 = v.T.dot(ax[:space])

        xt = ax0[:nroots] - e[:, None] * x0[:nroots]
        ax0 = None

        dx_norm = np.linalg.norm(xt, axis=1)
        conv =  (abs(de) < tol) & (dx_norm < toloose)
        for k, ek in enumerate(e):
            if conv[k] and not conv_last[k]:
                log.debug('root %d converged  |r|= %4.3g  e= %s  max|de|= %4.3g',
                          k, dx_norm[k], ek, de[k])
        ax0 = None
        max_dx_norm = max(dx_norm)
        max_de = max(abs(de))
        if all(conv):
            log.debug('converged %d %d  |r|= %4.3g  e= %s  max|de|= %4.3g',
                      icyc, space, max_dx_norm, e, max_de)
            break

        mask = (~conv) & (dx_norm**2 > lindep)
        xt = precond(xt[mask], e[0], x0[mask])
        valid_xs = xs[:space]
        for _ in range(2):
            xt -= np.dot(np.dot(xt, valid_xs.T.conj()), valid_xs)
        xt_norm = np.linalg.norm(xt, axis=1)
        keep_mask = (xt_norm**2 > lindep)
        xt = xt[keep_mask]
        xt_norm = xt_norm[keep_mask]

        if len(xt)==0:
            log.debug('Linear dependency in trial subspace. |r| for each state %s',
                      dx_norm)
            conv = dx_norm < toloose
            break
        log.debug('davidson %d %d  |r|= %4.3g  e= %s  max|de|= %4.3g  lindep= %4.3g',
                  icyc, space, max_dx_norm, e, max_de, np.linalg.norm(xt, axis=1).min())

        xt /= xt_norm[:, None]

        fresh_start = space+len(xt) > max_space
        if callable(callback):
            callback(locals())

    return conv, e, x0


def eigh(aop, x0, precond, tol_residual=1e-5, lindep=1e-12, nroots=1,
         x0sym=None, pick=None, max_cycle=50, max_memory=MAX_MEMORY,
         verbose=logger.WARN):
    '''
    slightly modified from pyscf.tdscf._lr_eig.eigh
    '''

    assert callable(pick)
    assert callable(precond)

    log = logger.new_logger(verbose)

    if isinstance(x0, np.ndarray) and x0.ndim == 1:
        x0 = x0[None,:]
    x0 = np.asarray(x0)

    x0_size = x0.shape[1]
    if MAX_SPACE_INC is None:
        space_inc = nroots
    else:
        # Adding too many trial bases in each iteration may cause larger errors
        space_inc = min(nroots, min(MAX_SPACE_INC, x0_size//2))

    max_space = int(max_memory*1e6/8/x0_size / 2 - nroots - space_inc)
    if max_space < nroots * 4 < x0_size:
        log.warn('Not enough memory to store trial space in _lr_eig.eigh')
    max_space = max(max_space, nroots * 4)
    max_space = min(max_space, x0_size)
    log.debug(f'Set max_space {max_space}, space_inc {space_inc}')

    xs = np.zeros((0, x0_size))
    ax = np.zeros((0, x0_size))
    e = w = v = None
    conv_last = conv = np.zeros(nroots, dtype=bool)
    xt = x0

    if x0sym is not None:
        xt_ir = np.asarray(x0sym)
        xs_ir = np.array([], dtype=xt_ir.dtype)

    for icyc in range(max_cycle):
        xt, xt_idx = _qr(xt, lindep)
        # Generate at most space_inc trial vectors
        if icyc != 0:
            xt = xt[:space_inc]
            xt_idx = xt_idx[:space_inc]

        row0 = len(xs)
        axt = aop(xt)
        xs = np.vstack([xs, xt])
        ax = np.vstack([ax, axt])
        if x0sym is not None:
            xs_ir = np.hstack([xs_ir, xt_ir[xt_idx]])

        # Compute heff = xs.conj().dot(ax.T)
        if w is None:
            heff = xs.conj().dot(ax.T)
        else:
            hsub = xt.conj().dot(ax.T)
            heff = np.block([[np.diag(w), hsub[:,:row0].conj().T],
                             [hsub[:,:row0], hsub[:,row0:]]])

        if x0sym is None:
            w, v = scipy.linalg.eigh(heff)
        else:
            # Diagonalize within eash symmetry sectors
            row1 = len(xs)
            w = np.empty(row1)
            v = np.zeros((row1, row1))
            v_ir = []
            i1 = 0
            for ir in set(xs_ir):
                idx = np.where(xs_ir == ir)[0]
                i0, i1 = i1, i1 + idx.size
                w_sub, v_sub = scipy.linalg.eigh(heff[idx[:,None],idx])
                w[i0:i1] = w_sub
                v[idx,i0:i1] = v_sub
                v_ir.append([ir] * idx.size)
            w_idx = np.argsort(w)
            w = w[w_idx]
            v = v[:,w_idx]
            xs_ir = np.hstack(v_ir)[w_idx]

        w, v, idx = pick(w, v, nroots, locals())
        if x0sym is not None:
            xs_ir = xs_ir[idx]
        if len(w) == 0:
            raise RuntimeError('Not enough eigenvalues')

        e, elast = w[:nroots], e
        if elast is None:
            de = e
        elif elast.size != e.size:
            log.debug('Number of roots different from the previous step (%d,%d)',
                      e.size, elast.size)
            de = e
        else:
            # mapping to previous eigenvectors
            vlast = np.eye(nroots)
            elast, conv_last = _sort_elast(elast, conv, vlast,
                                           v[:nroots,:nroots], log)
            de = e - elast

        xs = v.T.dot(xs)
        ax = v.T.dot(ax)
        if len(xs) * 2 > max_space:
            row0 = max(nroots, max_space-space_inc)
            xs = xs[:row0]
            ax = ax[:row0]
            w = w[:row0]
            if x0sym is not None:
                xs_ir = xs_ir[:row0]

        t_size = max(nroots, max_space-len(xs))
        xt = -w[:t_size,None] * xs[:t_size]
        xt += ax[:t_size]
        if x0sym is not None:
            xt_ir = xs_ir[:t_size]

        dx_norm = np.linalg.norm(xt, axis=1)
        max_dx_norm = max(dx_norm[:nroots])
        conv = dx_norm[:nroots] < tol_residual
        for k, ek in enumerate(e[:nroots]):
            if conv[k] and not conv_last[k]:
                log.debug('root %d converged  |r|= %4.3g  e= %s  max|de|= %4.3g',
                          k, dx_norm[k], ek, de[k])
            else:
                log.debug1('root %d  |r|= %4.3g  e= %s  max|de|= %4.3g',
                          k, dx_norm[k], ek, de[k])
        ide = np.argmax(abs(de))
        if all(conv):
            log.debug('converged %d %d  |r|= %4.3g  e= %s  max|de|= %4.3g',
                      icyc, len(xs), max_dx_norm, e, de[ide])
            break

        # remove subspace linear dependency
        for k, xk in enumerate(xt):
            if dx_norm[k] > tol_residual:
                xt[k] = precond(xk, e[0])
        for _ in range(2):
            xt -= xs.conj().dot(xt.T).T.dot(xs)
        xt_norm = np.linalg.norm(xt, axis=1)

        remaining = []
        for k, xk in enumerate(xt):
            if dx_norm[k] > tol_residual and xt_norm[k]**2 > lindep:
                xt[k] /= xt_norm[k]
                remaining.append(k)
        if len(remaining) == 0:
            log.debug(f'Linear dependency in trial subspace. |r| for each state {dx_norm}')
            break

        xt = xt[remaining]
        log.debug1('Generate %d trial vectors. Drop %d vectors',
                   len(xt), dx_norm.size - len(xt))

        if x0sym is not None:
            xt_ir = xt_ir[remaining]
        norm_min = xt_norm[remaining].min()
        log.debug('davidson %d %d |r|= %4.3g  e= %s  max|de|= %4.3g  lindep= %4.3g',
                  icyc, len(xs), max_dx_norm, e, de[ide], norm_min)

    x0 = xs[:nroots]
    # Check whether the solver finds enough eigenvectors.
    if len(x0) < min(x0_size, nroots):
        log.warn(f'Not enough eigenvectors (len(x0)={len(x0)}, nroots={nroots})')

    return conv, e, x0

def _qr(xs, lindep=1e-12):
    """
    Orthogonalize trial vectors in input order.
    Return orthonormal vectors and their original indices.
    """
    xs = np.array(xs, dtype=np.result_type(xs, np.float64), copy=True)
    nv = 0
    idx = []

    for i in range(len(xs)):
        xi = xs[i]
        if nv > 0:
            prod = xs[:nv].conj().dot(xi)
            xi -= prod.dot(xs[:nv])
            prod = xs[:nv].conj().dot(xi)
            xi -= prod.dot(xs[:nv])

        norm2 = np.vdot(xi, xi).real
        if norm2 > lindep:
            xs[nv] = xi / np.sqrt(norm2)
            idx.append(i)
            nv += 1
    return xs[:nv], np.asarray(idx, dtype=int)
