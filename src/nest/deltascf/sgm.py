#!/usr/bin/env python
# Copyright 2026 The NEST Developers. All Rights Reserved.
# Copyright 2014-2021 The PySCF Developers. All Rights Reserved.
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
#
# Author: Tai Wang <wtpeter@pku.edu.cn> & Codex
# Ref: JCTC 2020, 16, 1699

from functools import reduce

import numpy
import scipy.linalg

from pyscf import __config__, lib
from pyscf.lib import logger
from pyscf.scf import addons, hf, hf_symm


_ARMIJO_C1 = 1e-4
_MIN_ALPHA = 1e-12
_CURVATURE_TOL = 1e-10
_PRECOND_FLOOR = 1e-12
_MAX_PRECOND = 1e4


# Adapted from PySCF PR #3448, commit e45d0fe9c201825dd6b7069ade7d6532b9952617.
# https://github.com/pyscf/pyscf/pull/3448
def gen_g_hop_rohf(mf, mo_coeff, mo_occ, fock_ao=None, h1e=None,
                   with_symmetry=True):
    '''ROHF orbital gradient and symmetric orbital Hessian action.'''
    mo_occ = numpy.asarray(mo_occ)
    mol = mf.mol
    mo_coeff0 = mo_coeff
    if getattr(mf, '_scf', None) and mf._scf.mol != mol:
        # TODO: construct vind with dual-basis treatment
        mo_coeff = addons.project_mo_nr2nr(mf._scf.mol, mo_coeff, mol)

    if getattr(fock_ao, 'focka', None) is None:
        if getattr(mf, '_scf', None) and mf._scf.mol != mol:
            h1e = mf.get_hcore(mol)
        dm0 = mf.make_rdm1(mo_coeff, mo_occ)
        fock_ao = mf.get_fock(h1e, dm=dm0)
        focka = reduce(numpy.dot, (mo_coeff.conj().T, fock_ao.focka, mo_coeff))
        fockb = reduce(numpy.dot, (mo_coeff.conj().T, fock_ao.fockb, mo_coeff))
    else:
        focka = reduce(numpy.dot, (mo_coeff0.conj().T, fock_ao.focka, mo_coeff0))
        fockb = reduce(numpy.dot, (mo_coeff0.conj().T, fock_ao.fockb, mo_coeff0))
    mo_occa = occidxa = mo_occ > 0
    mo_occb = occidxb = mo_occ == 2
    viridxa = ~occidxa
    viridxb = ~occidxb
    uniq_var_a = viridxa[:,None] & occidxa
    uniq_var_b = viridxb[:,None] & occidxb
    uniq_ab = uniq_var_a | uniq_var_b
    nmo = mo_coeff.shape[-1]
    orboa = mo_coeff[:,occidxa]
    orbob = mo_coeff[:,occidxb]
    orbva = mo_coeff[:,viridxa]
    orbvb = mo_coeff[:,viridxb]

    g = numpy.zeros_like(focka)
    g[uniq_var_a] = focka[uniq_var_a]
    g[uniq_var_b] += fockb[uniq_var_b]
    g = g[uniq_ab]
    ea = focka.diagonal().real
    eb = fockb.diagonal().real
    h_diag = numpy.zeros_like(focka.real)
    h_diag[uniq_var_a] = (ea[:,None] - ea)[uniq_var_a]
    h_diag[uniq_var_b] += (eb[:,None] - eb)[uniq_var_b]
    h_diag = h_diag[uniq_ab]

    if with_symmetry and mol.symmetry:
        orbsym = hf_symm.get_orbsym(mol, mo_coeff)
        sym_forbid = (orbsym[:,None] != orbsym)[uniq_ab]
        g[sym_forbid] = 0
        h_diag[sym_forbid] = 0

    gmat = hf.unpack_uniq_var(g, mo_occ)
    vind = mf.gen_response((mo_coeff,)*2, (mo_occa, mo_occb),
                           hermi=1, with_nlc=False)

    def h_op(x):
        if with_symmetry and mol.symmetry:
            x = x.copy()
            x[sym_forbid] = 0
        x1 = numpy.zeros((nmo,nmo), dtype=x.dtype)
        x1[uniq_ab] = x
        x1a = x1[uniq_var_a].reshape(orbva.shape[1], orboa.shape[1])
        x1b = x1[uniq_var_b].reshape(orbvb.shape[1], orbob.shape[1])
        d1a = reduce(numpy.dot, (orbva, x1a, orboa.conj().T))
        d1b = reduce(numpy.dot, (orbvb, x1b, orbob.conj().T))
        dm1 = numpy.array((d1a+d1a.conj().T, d1b+d1b.conj().T))
        v1a, v1b = vind(dm1)

        # Keep the shared rotation's oo/vv contributions for each spin.
        kappa = hf.unpack_uniq_var(x, mo_occ)
        hmat_a = focka.dot(kappa) - kappa.dot(focka)
        hmat_b = fockb.dot(kappa) - kappa.dot(fockb)
        hmat_a += reduce(numpy.dot, (mo_coeff.conj().T, v1a, mo_coeff))
        hmat_b += reduce(numpy.dot, (mo_coeff.conj().T, v1b, mo_coeff))
        hx = numpy.zeros_like(hmat_a)
        hx[uniq_var_a] = hmat_a[uniq_var_a]
        hx[uniq_var_b] += hmat_b[uniq_var_b]
        # Transport the moving-frame gradient back to the reference frame.
        hx += .5 * (kappa.dot(gmat) - gmat.dot(kappa))
        hx = hx[uniq_ab]
        if with_symmetry and mol.symmetry:
            hx[sym_forbid] = 0
        return hx

    return g, h_op, h_diag


class LBFGSHistory:
    '''Bounded L-BFGS history for inverse-Hessian two-loop recursion.'''

    def __init__(self, max_size=8, min_curvature=1e-10):
        self.max_size = max_size
        self.min_curvature = min_curvature
        self.s_list = []
        self.y_list = []
        self.rho_list = []

    def __len__(self):
        return len(self.s_list)

    def clear(self):
        self.s_list.clear()
        self.y_list.clear()
        self.rho_list.clear()

    def push(self, step, grad_diff):
        sy = numpy.dot(step, grad_diff)
        # A relative test retains valid secant pairs as the gradients shrink.
        scale = numpy.linalg.norm(step) * numpy.linalg.norm(grad_diff)
        if sy <= self.min_curvature * scale:
            return False, sy

        self.s_list.append(step.copy())
        self.y_list.append(grad_diff.copy())
        self.rho_list.append(1.0 / sy)

        if len(self.s_list) > self.max_size:
            self.s_list.pop(0)
            self.y_list.pop(0)
            self.rho_list.pop(0)

        return True, sy

    def direction(self, grad, h0_inv):
        '''Return p = -H_k^{-1} grad.'''
        if not self.s_list:
            return -(h0_inv * grad)

        nvec = len(self.s_list)
        alpha = numpy.empty(nvec)
        q = grad.copy()

        for i in range(nvec - 1, -1, -1):
            alpha[i] = self.rho_list[i] * numpy.dot(self.s_list[i], q)
            q -= alpha[i] * self.y_list[i]

        z = h0_inv * q

        for i in range(nvec):
            beta = self.rho_list[i] * numpy.dot(self.y_list[i], z)
            z += (alpha[i] - beta) * self.s_list[i]

        return -z


def _somo_overlaps(mo_ref, mo_coeff, mo_occ, s1e):
    '''Cosines of principal angles between the two SOMO subspaces.'''
    somo = numpy.asarray(mo_occ) == 1
    overlap = reduce(numpy.dot, (mo_ref[:, somo].conj().T, s1e,
                                mo_coeff[:, somo]))
    return numpy.linalg.svd(overlap, compute_uv=False)


class SGM(lib.StreamObject):
    '''Square Gradient Minimization mixin for ROHF-like SCF objects.

    Important parameters:
        tol
            Convergence threshold on sqrt(Delta), where Delta = g_orb.T@g_orb.
            An explicitly set PySCF conv_tol_grad takes precedence over tol.
        gradient_scale
            Scalar c from the SGM paper (default here: 0.75).
            It scales the gradient seen by L-BFGS and the L-BFGS
            y_k history; the line search still uses the true Delta directional
            derivative.
        somo_overlap_tol
            Warn if the smallest SOMO overlap singular value falls below
            this threshold relative to the initial or previous orbitals.
            Default 0.7 corresponds to a largest principal angle of about
            46 degrees. This is a diagnostic, not a state constraint.

    The returned object also inherits the input ROHF/ROKS class. Use
    mf.SGM().set(...).run(mo_coeff, mo_occ), or SGM(mf).kernel(...).
    NLC response is omitted. Only real orbital rotations are supported.
    '''

    __name_mixin__ = 'SGM'

    max_cycle = getattr(__config__, 'sgm_max_cycle', 200)
    tol = getattr(__config__, 'sgm_tol', 1e-4)
    lbfgs_memory = getattr(__config__, 'sgm_lbfgs_memory', 8)
    gradient_scale = getattr(__config__, 'sgm_gradient_scale', 0.75)
    somo_overlap_tol = getattr(__config__, 'sgm_somo_overlap_tol', 0.7)
    canonicalization = getattr(__config__,
                               'soscf_newton_ah_SOSCF_canonicalization', True)

    _keys = {'max_cycle', 'tol', 'lbfgs_memory', 'gradient_scale',
             'canonicalization', 'somo_overlap_tol'}

    gen_g_hop = staticmethod(gen_g_hop_rohf)

    def __new__(cls, mf):
        if isinstance(mf, SGM):
            return mf
        if not isinstance(mf, hf.SCF) or not mf.istype('ROHF'):
            raise NotImplementedError('SGM currently supports ROHF/ROKS objects')
        obj = object.__new__(cls)
        return lib.set_class(obj, (cls, mf.__class__))

    def __init__(self, mf):
        if mf is self or isinstance(mf, SGM):
            return
        self.__dict__.update(mf.__dict__)
        self._scf = mf

    def check_sanity(self):
        if not numpy.isfinite(self.gradient_scale) or self.gradient_scale <= 0:
            raise ValueError('gradient_scale must be finite and positive')
        tol = self.tol if self.conv_tol_grad is None else self.conv_tol_grad
        if not numpy.isfinite(tol) or tol <= 0:
            raise ValueError('SGM gradient tolerance must be finite and positive')
        super().check_sanity()
        return self

    def dump_flags(self, verbose=None):
        super().dump_flags(verbose)
        log = logger.new_logger(self, verbose)
        log.info('SGM gradient tolerance = %g',
                 self.tol if self.conv_tol_grad is None else self.conv_tol_grad)
        log.info('SGM gradient scale = %g', self.gradient_scale)
        log.info('SGM L-BFGS memory = %d', self.lbfgs_memory)
        log.info('SGM SOMO overlap warning threshold = %g', self.somo_overlap_tol)
        log.info('SGM canonicalization = %s', self.canonicalization)
        return self

    def _preconditioner(self, h_diag):
        # Diagonal, frozen-Fock approximation to 2*J.T@J for Delta = g.T@g.
        # J = d g / d kappa; this is not the full Delta Hessian away from g=0.
        denom = numpy.maximum(2.0 * h_diag ** 2, _PRECOND_FLOOR)
        return numpy.minimum(1.0 / denom, _MAX_PRECOND)

    def _trial_mo(self, mo_coeff, mo_occ, step):
        kappa = hf.unpack_uniq_var(step, mo_occ)
        mo_trial = numpy.dot(mo_coeff, scipy.linalg.expm(kappa))
        if self.mol.symmetry:
            orbsym = hf_symm.get_orbsym(self.mol, mo_coeff)
            mo_trial = lib.tag_array(mo_trial, orbsym=orbsym)
        return mo_trial

    def _exact_sgm_state(self, mo_coeff, mo_occ, fock_ao=None, h1e=None):
        g_orb, h_op, h_diag = self.gen_g_hop(self, mo_coeff, mo_occ, fock_ao, h1e)
        delta = numpy.dot(g_orb, g_orb)
        grad_delta = 2.0 * h_op(g_orb)
        return h_diag, delta, grad_delta

    def _line_search(self, mo_coeff, mo_occ, direction, delta, grad_delta,
                     h1e, s1e):
        alpha = 1.0
        dir_deriv = numpy.dot(direction, grad_delta)

        while alpha >= _MIN_ALPHA:
            step = alpha * direction
            mo_trial = self._trial_mo(mo_coeff, mo_occ, step)

            dm_trial = self.make_rdm1(mo_trial, mo_occ)
            vhf_trial = self.get_veff(self.mol, dm_trial)
            fock_trial = self.get_fock(h1e, s1e, vhf_trial, dm_trial)
            g_trial = self.get_grad(mo_trial, mo_occ, fock_trial)
            delta_trial = numpy.dot(g_trial, g_trial)

            if delta_trial <= delta + _ARMIJO_C1 * alpha * dir_deriv:
                return alpha, step, mo_trial, fock_trial, vhf_trial

            alpha *= 0.5

        return None

    def _descent_direction(self, history, grad_delta, opt_grad_delta, h0_inv,
                           log, cycle):
        direction = history.direction(opt_grad_delta, h0_inv)
        dir_deriv = numpy.dot(direction, grad_delta)

        if dir_deriv < 0:
            return direction

        log.warn('SGM: non-descent L-BFGS direction at iter %d; reset history',
                 cycle)
        history.clear()

        direction = -(h0_inv * opt_grad_delta)
        dir_deriv = numpy.dot(direction, grad_delta)
        if dir_deriv >= 0:
            log.warn('SGM: fallback direction is not descent at iter %d', cycle)
            return None
        return direction

    def kernel(self, mo_coeff=None, mo_occ=None):
        if self.verbose >= logger.WARN:
            self.check_sanity()
        if self.verbose >= logger.INFO:
            self.dump_flags()
        log = logger.new_logger(self)
        if mo_coeff is None:
            mo_coeff = self.mo_coeff
        if mo_occ is None:
            mo_occ = self.mo_occ
        if mo_occ is None:
            raise RuntimeError('mo_occ must be specified for SGM')
        if mo_coeff is None:
            raise RuntimeError('mo_coeff must be specified for SGM')
        tol = self.tol if self.conv_tol_grad is None else self.conv_tol_grad

        mo_guess = mo_coeff.copy()
        h1e = self.get_hcore()
        s1e = self.get_ovlp()
        dm = self.make_rdm1(mo_coeff, mo_occ)
        vhf = self.get_veff(self.mol, dm)
        fock = self.get_fock(h1e, s1e, vhf, dm)
        e_tot = self.energy_tot(dm, h1e, vhf)

        t0 = (logger.process_clock(), logger.perf_counter())
        h_diag, delta, grad_delta = self._exact_sgm_state(mo_coeff, mo_occ, fock, h1e)
        opt_grad_delta = self.gradient_scale * grad_delta

        history = LBFGSHistory(self.lbfgs_memory, _CURVATURE_TOL)
        log.info('SGM: initial E = %17.12f  Delta = %.3e  |g| = %.3e',
                 e_tot, delta, numpy.sqrt(delta))

        self.cycles = 0
        for cycle in range(self.max_cycle):
            norm_g = numpy.sqrt(delta)
            if norm_g < tol:
                break

            h0_inv = self._preconditioner(h_diag)
            direction = self._descent_direction(history, grad_delta,
                                                opt_grad_delta, h0_inv, log,
                                                cycle)
            if direction is None:
                break

            trial = self._line_search(mo_coeff, mo_occ, direction,
                                      delta, grad_delta, h1e, s1e)
            if trial is None and len(history):
                log.warn('SGM: line search failed at iter %d; retry steepest '
                         'descent after clearing L-BFGS history', cycle)
                history.clear()
                direction = -(h0_inv * opt_grad_delta)
                trial = self._line_search(mo_coeff, mo_occ, direction,
                                          delta, grad_delta, h1e, s1e)

            if trial is None:
                log.warn('SGM: line search failed at iter %d', cycle)
                break

            alpha, step, mo_trial, fock, vhf_new = trial
            dm = self.make_rdm1(mo_trial, mo_occ)
            e_new = self.energy_tot(dm, h1e, vhf_new)

            h_diag_new, delta_new, grad_delta_new = self._exact_sgm_state(
                mo_trial, mo_occ, fock, h1e)
            opt_grad_delta_new = self.gradient_scale * grad_delta_new

            step_grad = opt_grad_delta_new - opt_grad_delta
            accepted, sy = history.push(step, step_grad)
            if not accepted:
                if sy < 0:
                    history.clear()
                log.debug1('SGM: skip L-BFGS pair at iter %d; sTy = %.4g',
                           cycle, sy)

            somo_initial = _somo_overlaps(mo_guess, mo_trial, mo_occ, s1e)
            somo_previous = _somo_overlaps(mo_coeff, mo_trial, mo_occ, s1e)
            if somo_initial.size:
                log.info('SGM SOMO overlap singular values: initial %s; previous %s',
                         somo_initial, somo_previous)
                if min(somo_initial[-1], somo_previous[-1]) < self.somo_overlap_tol:
                    log.warn('SGM: SOMO subspace deviation at iter %d: '
                             'min overlap initial = %.6f, previous = %.6f '
                             '(threshold %.3f)',
                             cycle, somo_initial[-1], somo_previous[-1], self.somo_overlap_tol)

            delta_e = e_new - e_tot
            mo_coeff = mo_trial
            h_diag = h_diag_new
            delta = delta_new
            grad_delta = grad_delta_new
            opt_grad_delta = opt_grad_delta_new
            e_tot = e_new
            vhf = vhf_new
            self.cycles = cycle + 1

            log.info('SGM iter %3d: E = %17.12f  dE = % .3e  '
                     'Delta = %.3e  |g| = %.3e  alpha = %.3f  '
                     '|step| = %.3e',
                     cycle, e_tot, delta_e, delta, numpy.sqrt(delta),
                     alpha, numpy.linalg.norm(step))
            if callable(self.callback):
                self.callback(locals())

        log.timer('SGM optimization', *t0)

        conv = numpy.sqrt(delta) < tol
        if conv:
            log.info('SGM converged in %d iterations: |g| = %.6g < %.6g',
                     self.cycles, numpy.sqrt(delta), tol)
        else:
            log.info('SGM did not converge in %d iterations: |g| = %.6g',
                     self.cycles, numpy.sqrt(delta))

        mo_energy, mo_coeff_canon = self.canonicalize(mo_coeff, mo_occ, fock)
        if self.canonicalization:
            log.info('Canonicalize SCF orbitals')
            mo_coeff = mo_coeff_canon

        self.mo_coeff = mo_coeff
        self.mo_occ = mo_occ
        self.converged = conv
        self.mo_energy = mo_energy
        self.e_tot = e_tot
        self._finalize()
        if self.chkfile:
            self.dump_chk(self.chkfile)
        return self.e_tot

    scf = kernel

    def undo_sgm(self):
        obj = lib.view(self, lib.drop_class(self.__class__, SGM))
        del obj._scf
        return obj


hf.SCF.SGM = lib.class_as_method(SGM)
