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

import io
import unittest
from unittest import mock
import numpy

from pyscf import gto, lib
from pyscf.lib import logger
from nest.soscf import sgm


# Q-Chem benchmark, HCHO Rydberg excitation:
#
# $rem
# METHOD BHHLYP
# BASIS cc-pvdz
# SYMMETRY false
# SYM_IGNORE true
# NO_REORIENT True
# XC_GRID 000075000302
# UNRESTRICTED false
# delta_scf true
# $end
#
# $delta_scf
#   triplet restricted
#   triplet_SCF_algorithm SGM
#   somo_1 8
#   somo_2 15
# $end
#
# Q-Chem output:
#   Restricted open-shell triplet state = -113.131417 Ha
#   SOMO(1): best initial orbital 8 overlap 0.977366
#   SOMO(2): best initial orbital 9 overlap 0.991702


class KnownValues(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        atom = '''
        C  0.00000000  0.00000000 -1.13947666
        O  0.00000000  0.00000000  1.14402883
        H  0.00000000  1.76627623 -2.23398653
        H  0.00000000 -1.76627623 -2.23398653
        '''
        cls.mol0 = gto.M(atom=atom, charge=0, spin=0, basis='cc-pvdz',
                         verbose=0, output='/dev/null')
        cls.mf0 = cls.mol0.RKS(xc='BHandHLYP')
        cls.mf0.grids.atom_grid = (75, 302)
        cls.mf0.kernel()

        setocc = cls.mf0.to_uks().mo_occ
        setocc[1][7] -= 1
        setocc[0][14] += 1
        cls.ro_occ = setocc[0] + setocc[1]

        cls.mol1 = gto.M(atom=atom, charge=0, spin=2, basis='cc-pvdz',
                         verbose=0, output='/dev/null')

    @classmethod
    def tearDownClass(cls):
        cls.mol0.stdout.close()
        cls.mol1.stdout.close()
        del cls.mol0, cls.mol1, cls.mf0, cls.ro_occ

    def test_delta_gradient(self):
        mf = self.mol1.ROKS(xc='BHandHLYP')
        mf.grids.atom_grid = (75, 302)
        mf.mo_coeff = self.mf0.mo_coeff.copy()
        mf.mo_occ = self.ro_occ.copy()

        opt = sgm.SGM(mf)
        _h_diag, _delta, grad_delta = opt._exact_sgm_state(mf.mo_coeff,
                                                           mf.mo_occ)
        rng = numpy.random.default_rng(12)
        direction = rng.normal(size=grad_delta.size)
        direction /= numpy.linalg.norm(direction)

        eps = 1e-4
        mo_plus = opt._trial_mo(mf.mo_coeff, mf.mo_occ, eps * direction)
        mo_minus = opt._trial_mo(mf.mo_coeff, mf.mo_occ, -eps * direction)
        g_plus = mf.get_grad(mo_plus, mf.mo_occ)
        g_minus = mf.get_grad(mo_minus, mf.mo_occ)
        fd = (numpy.dot(g_plus, g_plus) - numpy.dot(g_minus, g_minus)) / (2*eps)
        analytic = numpy.dot(grad_delta, direction)
        scale = max(abs(fd), abs(analytic), 1e-12)

        self.assertLess(abs(fd - analytic) / scale, 1e-6,
                        'fd = %.12e, analytic = %.12e' % (fd, analytic))

    def test_hcho_rydberg_triplet(self):
        for tol, max_cycle in ((1e-4, 80), (1e-7, 120)):
            with self.subTest(tol=tol):
                mf = self.mol1.ROKS(xc='BHandHLYP')
                mf.grids.atom_grid = (75, 302)
                mf.mo_coeff = self.mf0.mo_coeff.copy()
                mf.mo_occ = self.ro_occ.copy()

                opt = mf.SGM().set(tol=tol, max_cycle=max_cycle)
                e_tot = opt.kernel()

                self.assertTrue(opt.converged)
                self.assertLess(numpy.linalg.norm(opt.get_grad(opt.mo_coeff, opt.mo_occ)), tol)
                self.assertLess(abs(e_tot - -113.131417), 1e-4)
                self.assertLess(abs(e_tot - -113.131420102713), 1e-6)


class OptimizerChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mol = gto.M(atom='O 0 0 0; H 0 0 1; H 0 1 0', basis='sto-3g',
                        spin=2, verbose=0)
        cls.mf = cls.mol.ROHF().run(conv_tol=1e-12)
        cls.occ = cls.mf.mo_occ.copy()
        opt = cls.mf.SGM()
        size = cls.mf.get_grad(cls.mf.mo_coeff, cls.occ).size
        step = numpy.random.default_rng(42).normal(size=size)*0.02
        cls.guess = opt._trial_mo(cls.mf.mo_coeff, cls.occ, step)

    def test_last_cycle_and_fock_reuse(self):
        opt = self.mf.SGM().set(max_cycle=1, canonicalization=False)
        h1e, s1e = opt.get_hcore(), opt.get_ovlp()
        hdiag, delta, grad = opt._exact_sgm_state(self.guess, self.occ)
        direction = -opt._preconditioner(hdiag)*opt.gradient_scale*grad
        trial = opt._line_search(self.guess, self.occ, direction, delta, grad, h1e, s1e)
        self.assertIsNotNone(trial)
        alpha, _, cnew, fock, _ = trial
        final_norm = numpy.linalg.norm(opt.get_grad(cnew, self.occ, fock))
        self.assertLess(final_norm, numpy.sqrt(delta))
        opt.tol = (numpy.sqrt(delta)+final_norm)/2
        trace = []
        opt.callback = lambda env: trace.append(env['delta'])
        with mock.patch.object(opt, 'get_veff', wraps=opt.get_veff) as veff:
            opt.kernel(self.guess, self.occ)
        self.assertTrue(opt.converged)
        self.assertEqual(opt.cycles, 1)
        self.assertEqual(len(trace), 1)
        # One initial build and one per line-search trial, none for the
        # accepted-point energy or analytic gradient.
        self.assertEqual(veff.call_count, 2+round(-numpy.log2(alpha)))
        numpy.testing.assert_allclose(opt.mo_coeff, cnew, atol=1e-12)

    def test_pyscf_interfaces(self):
        for mf in (self.mf, self.mol.ROKS(xc='BHandHLYP')):
            opt = mf.SGM().set(max_cycle=0, canonicalization=False)
            self.assertIsInstance(opt, lib.StreamObject)
            self.assertIsInstance(opt, mf.__class__)
            self.assertIs(sgm.SGM(opt), opt)
            self.assertIs(opt.SGM(), opt)
            self.assertIs(opt.scf.__func__, opt.kernel.__func__)
            self.assertIs(opt.run(self.guess, self.occ), opt)
            plain = opt.undo_sgm()
            self.assertIsInstance(plain, mf.__class__)
            self.assertNotIsInstance(plain, sgm.SGM)
            numpy.testing.assert_allclose(plain.mo_coeff, opt.mo_coeff)
        opt = self.mf.SGM().set(max_cycle=0, tol=1e-15, conv_tol_grad=100)
        opt.scf(self.guess, self.occ)
        self.assertTrue(opt.converged)

    def test_somo_subspace(self):
        s = numpy.diag([1., 2., 3., 4.])
        c = numpy.diag(1/numpy.sqrt(s.diagonal()))
        occ = numpy.array([2,1,1,0])
        theta = 0.83
        u = numpy.array([[numpy.cos(theta), -numpy.sin(theta)],
                         [numpy.sin(theta), numpy.cos(theta)]])
        rotated = c.copy()
        rotated[:,1:3] = c[:,1:3]@u
        numpy.testing.assert_allclose(sgm._somo_overlaps(c,rotated,occ,s), 1, atol=1e-14)
        leaked = c.copy()
        leaked[:,2] = numpy.cos(theta)*c[:,2]+numpy.sin(theta)*c[:,3]
        numpy.testing.assert_allclose(sgm._somo_overlaps(c,leaked,occ,s),
                                      [1,numpy.cos(theta)], atol=1e-14)
        leaked[:,1:3] = leaked[:,1:3]@u
        numpy.testing.assert_allclose(sgm._somo_overlaps(rotated,leaked,occ,s),
                                      [1,numpy.cos(theta)], atol=1e-14)

    def test_lbfgs_curvature_scale(self):
        for scale in (1., 1e-8):
            history = sgm.LBFGSHistory()
            step = scale*numpy.array([1., 2.])
            diff = scale*numpy.array([2., 3.])
            self.assertTrue(history.push(step, diff)[0])
            self.assertFalse(history.push(step, -diff)[0])
            self.assertFalse(history.push(step, numpy.zeros(2))[0])
            direction = history.direction(diff, numpy.ones(2))
            self.assertLess(numpy.dot(direction, diff), 0)

    def test_somo_warning_each_step(self):
        opt = self.mf.SGM().set(max_cycle=2, tol=1e-12, somo_overlap_tol=0.999999999,
                                verbose=logger.WARN, stdout=io.StringIO())
        opt.kernel(self.guess,self.occ)
        self.assertEqual(opt.cycles, 2)
        output = opt.stdout.getvalue()
        self.assertIn('SOMO subspace deviation at iter 0', output)
        self.assertIn('SOMO subspace deviation at iter 1', output)

    def test_frozen_fock_preconditioner(self):
        # Remove density response and off-diagonal Fock terms to isolate
        # the scale of the mean-field preconditioner in all RO blocks.
        opt = self.mf.SGM()
        nmo = len(self.occ)
        c = numpy.eye(nmo)
        fa = numpy.diag(numpy.linspace(-2,3,nmo))
        fb = numpy.diag(numpy.linspace(-1,4,nmo)**3)
        fock = lib.tag_array((fa+fb)/2, focka=fa, fockb=fb)
        _, _, hdiag = opt.gen_g_hop(opt,c,self.occ,fock)
        eps = 1e-4
        curvature = []
        for i in range(hdiag.size):
            step = numpy.zeros(hdiag.size)
            step[i] = eps
            gp = opt.get_grad(opt._trial_mo(c,self.occ,step), self.occ, fock)
            gm = opt.get_grad(opt._trial_mo(c,self.occ,-step), self.occ, fock)
            curvature.append((gp@gp+gm@gm)/eps**2)
        numpy.testing.assert_allclose(curvature,1/opt._preconditioner(hdiag),rtol=1e-6)

    def test_analytic_delta_curvature(self):
        for xc in (None, 'LDA,VWN', 'BHandHLYP'):
            with self.subTest(xc=xc):
                mf = self.mol.ROHF() if xc is None else self.mol.ROKS(xc=xc)
                if xc is not None:
                    mf.grids.level = 1
                opt = mf.SGM()
                # A nonstationary reference exercises the g.T@g'' term,
                # including third XC derivatives for LDA/GGA.
                c = self.guess
                g, hop, _ = opt.gen_g_hop(mf, c, self.occ)
                curvature = sgm.gen_delta_curvature_rohf(mf, c, self.occ)
                rng = numpy.random.default_rng(23)
                for _ in range(2):
                    x = rng.normal(size=g.size)
                    x /= numpy.linalg.norm(x)
                    exact, gn = curvature(x)
                    self.assertAlmostEqual(gn, 2*numpy.linalg.norm(hop(x))**2, places=9)
                    eps = 5e-4
                    plus = opt.get_grad(opt._trial_mo(c, self.occ, eps*x), self.occ)
                    minus = opt.get_grad(opt._trial_mo(c, self.occ, -eps*x), self.occ)
                    fd = (plus@plus + minus@minus - 2*g@g)/eps**2
                    self.assertLess(abs(fd-exact)/max(1., abs(exact)), 2e-6)

    def test_delta_curvature_at_stationary_point(self):
        c = self.mf.mo_coeff
        g, hop, _ = self.mf.SGM().gen_g_hop(self.mf, c, self.occ)
        curvature = sgm.gen_delta_curvature_rohf(self.mf, c, self.occ)
        x = numpy.random.default_rng(12).normal(size=g.size)
        x /= numpy.linalg.norm(x)
        exact, gn = curvature(x)
        self.assertLess(numpy.linalg.norm(g), 1e-6)
        self.assertLess(abs(exact-gn)/max(1., gn), 1e-6)


if __name__ == '__main__':
    print('Full tests for soscf.sgm')
    unittest.main()
