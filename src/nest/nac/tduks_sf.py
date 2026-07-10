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
#
# Ref:
# J. Chem. Theory Comput. 2026
#

import copy
from functools import reduce

import numpy as np

from pyscf import lib
from pyscf.lib import logger
from pyscf.scf import ucphf
from pyscf.dft import numint2c
from pyscf.grad import tdrhf as tdrhf_grad
from pyscf.grad import tdrks as tdrks_grad
from pyscf.grad import tduks as tduks_grad
from nest.sftda.numint2c_sftd import mcfun_eval_xc_adapter_sf


def get_Hellmann_Feymann(td_grad, x_y_I, x_y_J, atmlst=None, max_memory=6000, verbose=logger.INFO,
                         state_I=None, state_J=None):
    """
    Electronic part of spin-flip TDA/TDDFT nuclear gradients.

    Args:
        td_grad : grad.tduks_sf.Gradients object.

        x_y : a two-element list of numpy arrays
            TDDFT X and Y amplitudes. If Y is set to 0, this function computes
            TDA energy gradients.
    """
    log = logger.new_logger(td_grad, verbose)
    time0 = logger.process_clock(), logger.perf_counter()

    mol = td_grad.mol
    mf = td_grad.base._scf
    mo_coeff = mf.mo_coeff
    mo_energy = mf.mo_energy
    mo_occ = mf.mo_occ
    occidxa = np.where(mo_occ[0] > 0)[0]
    occidxb = np.where(mo_occ[1] > 0)[0]
    viridxa = np.where(mo_occ[0] == 0)[0]
    viridxb = np.where(mo_occ[1] == 0)[0]
    nocca = len(occidxa)
    noccb = len(occidxb)
    nvira = len(viridxa)
    nvirb = len(viridxb)
    orboa = mo_coeff[0][:, occidxa]
    orbob = mo_coeff[1][:, occidxb]
    orbva = mo_coeff[0][:, viridxa]
    orbvb = mo_coeff[1][:, viridxb]
    nao = mo_coeff[0].shape[0]
    nmoa = nocca + nvira
    nmob = noccb + nvirb

    if td_grad.base.extype == 0:
        y_I, x_I = x_y_I
        y_J, x_J = x_y_J
        if not isinstance(x_I, np.ndarray):
            x_I = np.zeros((nocca, nvirb))
        if not isinstance(x_J, np.ndarray):
            x_J = np.zeros((nocca, nvirb))
    elif td_grad.base.extype == 1:
        x_I, y_I = x_y_I
        x_J, y_J = x_y_J
        if not isinstance(y_I, np.ndarray):
            y_I = np.zeros((noccb, nvira))
        if not isinstance(y_J, np.ndarray):
            y_J = np.zeros((noccb, nvira))

    dvva_IJ = lib.einsum('ia,ib->ab', y_I, y_J)
    dvvb_IJ = lib.einsum('ia,ib->ab', x_I, x_J)
    dooa_IJ = -lib.einsum('ia,ja->ij', x_I, x_J)
    doob_IJ = -lib.einsum('ia,ja->ij', y_I, y_J)
    dvva_JI = lib.einsum('ia,ib->ab', y_J, y_I)
    dvvb_JI = lib.einsum('ia,ib->ab', x_J, x_I)
    dooa_JI = -lib.einsum('ia,ja->ij', x_J, x_I)
    doob_JI = -lib.einsum('ia,ja->ij', y_J, y_I)

    dmzooa_IJ = reduce(np.dot, (orboa, dooa_IJ, orboa.T))
    dmzooa_IJ += reduce(np.dot, (orbva, dvva_IJ, orbva.T))
    dmzoob_IJ = reduce(np.dot, (orbob, doob_IJ, orbob.T))
    dmzoob_IJ += reduce(np.dot, (orbvb, dvvb_IJ, orbvb.T))
    dmzooa_JI = reduce(np.dot, (orboa, dooa_JI, orboa.T))
    dmzooa_JI += reduce(np.dot, (orbva, dvva_JI, orbva.T))
    dmzoob_JI = reduce(np.dot, (orbob, doob_JI, orbob.T))
    dmzoob_JI += reduce(np.dot, (orbvb, dvvb_JI, orbvb.T))

    dmx_I = reduce(np.dot, (orbvb, x_I.T, orboa.T))
    dmy_I = reduce(np.dot, (orbob, y_I, orbva.T))
    dmx_J = reduce(np.dot, (orbvb, x_J.T, orboa.T))
    dmy_J = reduce(np.dot, (orbob, y_J, orbva.T))
    dmt_I = dmx_I + dmy_I
    dmt_J = dmx_J + dmy_J

    dmzooa = (dmzooa_IJ + dmzooa_JI) * 0.5
    dmzoob = (dmzoob_IJ + dmzoob_JI) * 0.5

    ni = mf._numint
    ni.libxc.test_deriv_order(mf.xc, 3, raise_error=True)
    omega, alpha, hyb = ni.rsh_and_hybrid_coeff(mf.xc, mol.spin)

    f1vo_I, f1oo, vxc1, k1ao = _contract_xc_kernel(td_grad, mf.xc, dmt_I, dmt_J, (dmzooa, dmzoob), True, True, max_memory)
    f1vo_J, _, _, _ = _contract_xc_kernel(td_grad, mf.xc, dmt_J, dmt_I, (dmzooa, dmzoob), True, True, max_memory)

    with_k = ni.libxc.is_hybrid_xc(mf.xc)
    if with_k:
        vj0, vk0 = mf.get_jk(mol, (dmzooa, dmzoob), hermi=1)
        vk1_I, vk1_J = mf.get_k(mol, (dmt_I, dmt_J), hermi=0) * hyb
        vk0 = vk0 * hyb
        if omega != 0:
            vk0 += mf.get_k(mol, (dmzooa, dmzoob), hermi=1, omega=omega) * (alpha - hyb)
            vk1_I_omega, vk1_J_omega = mf.get_k(mol, (dmt_I, dmt_J), hermi=0, omega=omega)
            vk1_I += vk1_I_omega * (alpha - hyb)
            vk1_J += vk1_J_omega * (alpha - hyb)

        veff0doo = vj0[0] + vj0[1] - vk0 + f1oo[:, 0] + k1ao[:, 0]
        wvoa = reduce(np.dot, (orbva.T, veff0doo[0], orboa))
        wvob = reduce(np.dot, (orbvb.T, veff0doo[1], orbob))

        veff0mo_I = reduce(np.dot, (mo_coeff[1].T, f1vo_I[0] - vk1_I, mo_coeff[0]))
        veff0mo_J = reduce(np.dot, (mo_coeff[1].T, f1vo_J[0] - vk1_J, mo_coeff[0]))
        wvoa += lib.einsum('ac,ka->ck', veff0mo_I[noccb:, nocca:], x_J) * 0.5
        wvoa += lib.einsum('ac,ka->ck', veff0mo_J[noccb:, nocca:], x_I) * 0.5
        wvoa -= lib.einsum('jk,jc->ck', veff0mo_I[:noccb, :nocca], y_J) * 0.5
        wvoa -= lib.einsum('jk,jc->ck', veff0mo_J[:noccb, :nocca], y_I) * 0.5
        wvob += lib.einsum('ac,ka->ck', veff0mo_I.T[nocca:, noccb:], y_J) * 0.5
        wvob += lib.einsum('ac,ka->ck', veff0mo_J.T[nocca:, noccb:], y_I) * 0.5
        wvob -= lib.einsum('jk,jc->ck', veff0mo_I.T[:nocca, :noccb], x_J) * 0.5
        wvob -= lib.einsum('jk,jc->ck', veff0mo_J.T[:nocca, :noccb], x_I) * 0.5

    else:
        vj0 = mf.get_j(mol, (dmzooa, dmzoob), hermi=1)
        veff0doo = vj0[0] + vj0[1] + f1oo[:, 0] + k1ao[:, 0]
        wvoa = reduce(np.dot, (orbva.T, veff0doo[0], orboa))
        wvob = reduce(np.dot, (orbvb.T, veff0doo[1], orbob))

        veff0mo_I = reduce(np.dot, (mo_coeff[1].T, f1vo_I[0], mo_coeff[0]))
        veff0mo_J = reduce(np.dot, (mo_coeff[1].T, f1vo_J[0], mo_coeff[0]))
        wvoa += lib.einsum('ac,ka->ck', veff0mo_I[noccb:, nocca:], x_J) * 0.5
        wvoa += lib.einsum('ac,ka->ck', veff0mo_J[noccb:, nocca:], x_I) * 0.5
        wvoa -= lib.einsum('jk,jc->ck', veff0mo_I[:noccb, :nocca], y_J) * 0.5
        wvoa -= lib.einsum('jk,jc->ck', veff0mo_J[:noccb, :nocca], y_I) * 0.5
        wvob += lib.einsum('ac,ka->ck', veff0mo_I.T[nocca:, noccb:], y_J) * 0.5
        wvob += lib.einsum('ac,ka->ck', veff0mo_J.T[nocca:, noccb:], y_I) * 0.5
        wvob -= lib.einsum('jk,jc->ck', veff0mo_I.T[:nocca, :noccb], x_J) * 0.5
        wvob -= lib.einsum('jk,jc->ck', veff0mo_J.T[:nocca, :noccb], x_I) * 0.5

    vresp = mf.gen_response(hermi=1)

    def fvind(x):
        xa = x[0, : nvira * nocca].reshape(nvira, nocca)
        xb = x[0, nvira * nocca :].reshape(nvirb, noccb)
        dma = reduce(np.dot, (orbva, xa, orboa.T))
        dmb = reduce(np.dot, (orbvb, xb, orbob.T))
        dm1 = np.stack((dma + dma.T, dmb + dmb.T))
        v1 = vresp(dm1)
        v1a = reduce(np.dot, (orbva.T, v1[0], orboa))
        v1b = reduce(np.dot, (orbvb.T, v1[1], orbob))
        return np.hstack((v1a.ravel(), v1b.ravel()))

    z1a, z1b = ucphf.solve(
        fvind, mo_energy, mo_occ, (wvoa, wvob), max_cycle=td_grad.cphf_max_cycle, tol=td_grad.cphf_conv_tol
    )[0]
    time1 = log.timer('Z-vector using UCPHF solver', *time0)

    z1ao = np.empty((2, nao, nao))
    z1ao[0] = reduce(np.dot, (orbva, z1a, orboa.T))
    z1ao[1] = reduce(np.dot, (orbvb, z1b, orbob.T))
    veff = vresp((z1ao + z1ao.transpose(0, 2, 1)))

    im0a = np.zeros((nmoa, nmoa))
    im0b = np.zeros((nmob, nmob))
    im0a[:nocca, :nocca] = reduce(np.dot, (orboa.T, veff0doo[0] + veff[0], orboa))
    im0b[:noccb, :noccb] = reduce(np.dot, (orbob.T, veff0doo[1] + veff[1], orbob))
    im0a[:nocca, :nocca] += lib.einsum('al,ka->lk', veff0mo_I[noccb:, :nocca], x_J) * 0.5
    im0a[:nocca, :nocca] += lib.einsum('al,ka->lk', veff0mo_J[noccb:, :nocca], x_I) * 0.5

    im0b[:noccb, :noccb] += lib.einsum('al,ka->lk', veff0mo_I.T[nocca:, :noccb], y_J) * 0.5
    im0b[:noccb, :noccb] += lib.einsum('al,ka->lk', veff0mo_J.T[nocca:, :noccb], y_I) * 0.5

    im0a[nocca:, nocca:] = lib.einsum('jd,jc->dc', veff0mo_I[:noccb, nocca:], y_J) * 0.5
    im0a[nocca:, nocca:] += lib.einsum('jd,jc->dc', veff0mo_J[:noccb, nocca:], y_I) * 0.5

    im0b[noccb:, noccb:] = lib.einsum('jd,jc->dc', veff0mo_I.T[:nocca, noccb:], x_J) * 0.5
    im0b[noccb:, noccb:] += lib.einsum('jd,jc->dc', veff0mo_J.T[:nocca, noccb:], x_I) * 0.5

    im0a[:nocca, nocca:] = lib.einsum('jk,jc->kc', veff0mo_I[:noccb, :nocca], y_J)
    im0a[:nocca, nocca:] += lib.einsum('jk,jc->kc', veff0mo_J[:noccb, :nocca], y_I)
    im0b[:noccb, noccb:] = lib.einsum('jk,jc->kc', veff0mo_I.T[:nocca, :noccb], x_J)
    im0b[:noccb, noccb:] += lib.einsum('jk,jc->kc', veff0mo_J.T[:nocca, :noccb], x_I)

    zeta_a = (mo_energy[0][:, None] + mo_energy[0]) * 0.5
    zeta_b = (mo_energy[1][:, None] + mo_energy[1]) * 0.5
    zeta_a[nocca:, :nocca] = mo_energy[0][:nocca]
    zeta_b[noccb:, :noccb] = mo_energy[1][:noccb]
    zeta_a[:nocca, nocca:] = mo_energy[0][nocca:]
    zeta_b[:noccb, noccb:] = mo_energy[1][noccb:]
    dm1a = np.zeros((nmoa, nmoa))
    dm1b = np.zeros((nmob, nmob))
    dm1a[:nocca, :nocca] = (dooa_IJ + dooa_JI) * 0.5
    dm1b[:noccb, :noccb] = (doob_IJ + doob_JI) * 0.5
    dm1a[nocca:, nocca:] = (dvva_IJ + dvva_JI) * 0.5
    dm1b[noccb:, noccb:] = (dvvb_IJ + dvvb_JI) * 0.5
    dm1a[nocca:, :nocca] = z1a * 2
    dm1b[noccb:, :noccb] = z1b * 2

    im0a = reduce(np.dot, (mo_coeff[0], im0a + zeta_a * dm1a, mo_coeff[0].T))
    im0b = reduce(np.dot, (mo_coeff[1], im0b + zeta_b * dm1b, mo_coeff[1].T))
    im0 = im0a + im0b

    mf_grad = td_grad.base._scf.nuc_grad_method()
    hcore_deriv = mf_grad.hcore_generator(mol)
    s1 = mf_grad.get_ovlp(mol)

    dmz1dooa = 4 * z1ao[0] + 2 * dmzooa
    dmz1doob = 4 * z1ao[1] + 2 * dmzoob
    oo0a = reduce(np.dot, (orboa, orboa.T))
    oo0b = reduce(np.dot, (orbob, orbob.T))
    as_dm1 = (dmz1dooa + dmz1doob) * 0.5

    if with_k:
        dm = (oo0a, dmz1dooa + dmz1dooa.T, oo0b, dmz1doob + dmz1doob.T)
        vj, vk = td_grad.get_jk(mol, dm, hermi=1)
        vj = vj.reshape(2, 2, 3, nao, nao)
        vk = vk.reshape(2, 2, 3, nao, nao) * hyb
        vk1_all = -td_grad.get_k(mol, (dmt_I, dmt_I.T, dmt_J, dmt_J.T)) * hyb
        vk1_I = vk1_all[:2]
        vk1_J = vk1_all[2:]
        if omega != 0:
            vk += td_grad.get_k(mol, dm, omega=omega).reshape(2, 2, 3, nao, nao) * (
                alpha - hyb
            )
            vk1_all_omega = -td_grad.get_k(mol, (dmt_I, dmt_I.T, dmt_J, dmt_J.T), omega=omega)
            vk1_I += vk1_all_omega[:2] * (alpha - hyb)
            vk1_J += vk1_all_omega[2:] * (alpha - hyb)
        veff1 = vj[0] + vj[1] - vk
    else:
        dm = (oo0a, dmz1dooa + dmz1dooa.T, oo0b, dmz1doob + dmz1doob.T)
        vj = td_grad.get_j(mol, dm, hermi=1).reshape(2, 2, 3, nao, nao)
        veff1 = vj[0] + vj[1]
        veff1 = np.stack((veff1, veff1))

    fxcz1 = tduks_grad._contract_xc_kernel(td_grad, mf.xc, 2 * z1ao, None, False, False, max_memory)[0]
    veff1[:, 0] += vxc1[:, 1:]
    veff1[:, 1] += (f1oo[:, 1:] + fxcz1[:, 1:] + k1ao[:, 1:]) * 4
    veff1a, veff1b = veff1
    time1 = log.timer('2e AO integral derivatives', *time1)

    if atmlst is None:
        atmlst = range(mol.natm)
    offsetdic = mol.offset_nr_by_atom()
    de = np.zeros((len(atmlst), 3))

    for k, ia in enumerate(atmlst):
        shl0, shl1, p0, p1 = offsetdic[ia]

        h1ao = hcore_deriv(ia)
        de[k] = lib.einsum('xpq,pq->x', h1ao, as_dm1)

        de[k] -= lib.einsum('xpq,pq->x', s1[:, p0:p1], im0[p0:p1])
        de[k] -= lib.einsum('xqp,pq->x', s1[:, p0:p1], im0[:, p0:p1])

        de[k] += lib.einsum('xpq,pq->x', veff1a[0, :, p0:p1], dmz1dooa[p0:p1]) * 0.5
        de[k] += lib.einsum('xpq,pq->x', veff1b[0, :, p0:p1], dmz1doob[p0:p1]) * 0.5
        de[k] += lib.einsum('xpq,qp->x', veff1a[0, :, p0:p1], dmz1dooa[:, p0:p1]) * 0.5
        de[k] += lib.einsum('xpq,qp->x', veff1b[0, :, p0:p1], dmz1doob[:, p0:p1]) * 0.5
        de[k] += lib.einsum('xij,ij->x', veff1a[1, :, p0:p1], oo0a[p0:p1]) * 0.5
        de[k] += lib.einsum('xij,ij->x', veff1b[1, :, p0:p1], oo0b[p0:p1]) * 0.5

        if td_grad.base.collinear == 'mcol':
            de[k] += lib.einsum('xpq,pq->x', f1vo_I[1:, p0:p1], dmt_J[p0:p1])
            de[k] += lib.einsum('xpq,pq->x', f1vo_J[1:, p0:p1], dmt_I[p0:p1])
            de[k] += lib.einsum('xpq,pq->x', f1vo_I[1:, p0:p1], dmt_J.T[p0:p1])
            de[k] += lib.einsum('xpq,pq->x', f1vo_J[1:, p0:p1], dmt_I.T[p0:p1])

        if with_k:
            de[k] += lib.einsum('xpq,pq->x', vk1_I[0, :, p0:p1], dmt_J[p0:p1])
            de[k] += lib.einsum('xpq,pq->x', vk1_J[0, :, p0:p1], dmt_I[p0:p1])
            de[k] += lib.einsum('xpq,pq->x', vk1_I[1, :, p0:p1], dmt_J.T[p0:p1])
            de[k] += lib.einsum('xpq,pq->x', vk1_J[1, :, p0:p1], dmt_I.T[p0:p1])

    log.timer('TDUKS nuclear gradients', *time0)
    return de


def _contract_xc_kernel(td_grad, xc_code, dmvo_I, dmvo_J, dmoo=None, with_vxc=True, with_kxc=True, max_memory=2000):
    mol = td_grad.mol
    mf = td_grad.base._scf
    grids = mf.grids

    ni = mf._numint
    xctype = ni._xc_type(xc_code)

    mo_coeff = mf.mo_coeff
    mo_occ = mf.mo_occ
    nao = mo_coeff[0].shape[0]
    shls_slice = (0, mol.nbas)
    ao_loc = mol.ao_loc_nr()
    dmvo_I = (dmvo_I + dmvo_I.T) * 0.5
    dmvo_J = (dmvo_J + dmvo_J.T) * 0.5

    f1vo = np.zeros((4, nao, nao))
    deriv = 2
    if dmoo is not None:
        f1oo = np.zeros((2, 4, nao, nao))
    else:
        f1oo = None
    if with_vxc:
        v1ao = np.zeros((2, 4, nao, nao))
    else:
        v1ao = None
    if with_kxc:
        k1ao = np.zeros((2, 4, nao, nao))
        deriv = 3
    else:
        k1ao = None

    if xctype == 'HF':
        return f1vo, f1oo, v1ao, k1ao
    elif xctype == 'LDA':
        fmat_, ao_deriv = tdrks_grad._lda_eval_mat_, 1
    elif xctype == 'GGA':
        fmat_, ao_deriv = tdrks_grad._gga_eval_mat_, 2
    elif xctype == 'MGGA':
        fmat_, ao_deriv = tdrks_grad._mgga_eval_mat_, 2
        logger.warn(td_grad, 'TDUKS-MGGA Gradients may be inaccurate due to grids response')
    else:
        raise NotImplementedError(f'td-uks for functional {xc_code}')

    if td_grad.base.collinear == 'mcol':
        nimc = numint2c.NumInt2C()
        nimc.collinear = 'mcol'
        nimc.collinear_samples = td_grad.base.collinear_samples
        eval_xc_eff = mcfun_eval_xc_adapter_sf(nimc, xc_code)

    for ao, mask, weight, coords in ni.block_loop(mol, grids, nao, ao_deriv, max_memory):
        ao0 = ao[0] if xctype == 'LDA' else ao
        rho = (
            ni.eval_rho2(mol, ao0, mo_coeff[0], mo_occ[0], mask, xctype, with_lapl=False),
            ni.eval_rho2(mol, ao0, mo_coeff[1], mo_occ[1], mask, xctype, with_lapl=False),
        )
        if td_grad.base.collinear == 'mcol':
            rho_z = np.array([rho[0] + rho[1], rho[0] - rho[1]])
            fxc_sf, kxc_sf = eval_xc_eff(xc_code, rho_z, deriv, xctype=xctype)[2:4]
            kxc_sf = np.stack(
                (kxc_sf[:, :, 0] + kxc_sf[:, :, 1], kxc_sf[:, :, 0] - kxc_sf[:, :, 1]), axis=2
            )
            rho1_I = ni.eval_rho(mol, ao0, dmvo_I, mask, xctype, hermi=1, with_lapl=False)
            if xctype == 'LDA':
                rho1_I = rho1_I[np.newaxis]
            wv = lib.einsum('yg,xyg,g->xg', rho1_I, 2 * fxc_sf, weight)
            fmat_(mol, f1vo, ao, wv, mask, shls_slice, ao_loc)

            if with_kxc:
                rho1_J = ni.eval_rho(mol, ao0, dmvo_J, mask, xctype, hermi=1, with_lapl=False)
                if xctype == 'LDA':
                    rho1_J = rho1_J[np.newaxis]
                wv = lib.einsum('xg,yg,xyczg,g->czg', rho1_I, rho1_J, 2 * kxc_sf, weight)
                fmat_(mol, k1ao[0], ao, wv[0], mask, shls_slice, ao_loc)
                fmat_(mol, k1ao[1], ao, wv[1], mask, shls_slice, ao_loc)

        if dmoo is not None or with_vxc:
            vxc, fxc = ni.eval_xc_eff(xc_code, rho, deriv=2, spin=1)[1:3]

        if dmoo is not None:
            rho2 = np.asarray(
                (
                    ni.eval_rho(mol, ao0, dmoo[0], mask, xctype, hermi=1, with_lapl=False),
                    ni.eval_rho(mol, ao0, dmoo[1], mask, xctype, hermi=1, with_lapl=False),
                )
            )
            if xctype == 'LDA':
                rho2 = rho2[:, np.newaxis]
            wv = lib.einsum('axg,axbyg,g->byg', rho2, fxc, weight)
            fmat_(mol, f1oo[0], ao, wv[0], mask, shls_slice, ao_loc)
            fmat_(mol, f1oo[1], ao, wv[1], mask, shls_slice, ao_loc)

        if with_vxc:
            wv = vxc * weight
            fmat_(mol, v1ao[0], ao, wv[0], mask, shls_slice, ao_loc)
            fmat_(mol, v1ao[1], ao, wv[1], mask, shls_slice, ao_loc)

    f1vo[1:] *= -1
    if f1oo is not None:
        f1oo[:, 1:] *= -1
    if v1ao is not None:
        v1ao[:, 1:] *= -1
    if k1ao is not None:
        k1ao[:, 1:] *= -1
    return f1vo, f1oo, v1ao, k1ao


def nac_csf(td_grad, x_y_I, x_y_J, atmlst=None):
    '''
    Compute the CSF (Configuration State Function) contribution to non-adiabatic coupling vectors (NACVs)
    between two spin-flip excited states.

    This term arises from the explicit nuclear coordinate dependence of the electronic wavefunctions
    (i.e., the derivative of the CI coefficients), and is necessary for reconstructing the full NACV
    consistent with finite-difference calculations.

    Important: This term breaks translational invariance and is NOT ETF-corrected.
    Including it will make the total NACV inconsistent with momentum conservation,
    but essential for matching finite-difference benchmarks.

    Args:
        td_grad:
            Gradient object associated with SF-TDA or SF-TDDFT calculation.
            Must contain molecular structure and orbital information.
    Returns:
        numpy.ndarray:
            CSF contribution to the NAC vector between states I and J.
            Shape: (natm, 3), in atomic units.
            Not ETF-corrected — use with caution in dynamics simulations.

    Notes:
        - This term + `get_Hellmann_Feynman` = Full NACV (matches finite difference).
        - The name "CSF" refers to the fact that this term originates from the derivative
          of the CI coefficients in the configuration state function basis.
    '''
    mol = td_grad.mol
    mf = td_grad.base._scf

    mo_coeff = mf.mo_coeff
    mo_occ = mf.mo_occ
    occidxa = np.where(mo_occ[0] > 0)[0]
    occidxb = np.where(mo_occ[1] > 0)[0]
    viridxa = np.where(mo_occ[0] == 0)[0]
    viridxb = np.where(mo_occ[1] == 0)[0]
    nocca = len(occidxa)
    noccb = len(occidxb)
    nvira = len(viridxa)
    nvirb = len(viridxb)
    orboa = mo_coeff[0][:, occidxa]
    orbob = mo_coeff[1][:, occidxb]
    orbva = mo_coeff[0][:, viridxa]
    orbvb = mo_coeff[1][:, viridxb]
    mo_coeff[0].shape[0]

    nocca + nvira
    noccb + nvirb

    if td_grad.base.extype == 0:
        x_ab_I, y_ba_I = x_y_I
        x_ab_J, y_ba_J = x_y_J
        if not isinstance(y_ba_I, np.ndarray):
            y_ba_I = np.zeros((nocca, nvirb))
        if not isinstance(y_ba_J, np.ndarray):
            y_ba_J = np.zeros((nocca, nvirb))
        x_ab_I = x_ab_I.T
        x_ba_I = (-y_ba_I).T
        x_ab_J = x_ab_J.T
        x_ba_J = y_ba_J.T
        dvv_a_IJ = np.einsum('ai,bi->ab', x_ab_I, x_ab_J)

        dvv_b_IJ = np.einsum('ai,bi->ab', x_ba_I, x_ba_J)

        doo_b_IJ = np.einsum('ai,aj->ij', x_ab_I, x_ab_J)

        doo_a_IJ = np.einsum('ai,aj->ij', x_ba_I, x_ba_J)

        dmzoo_a_IJ = reduce(np.dot, (orboa, doo_a_IJ, orboa.T))

        dmzoo_b_IJ = reduce(np.dot, (orbob, doo_b_IJ, orbob.T))

        dmzoo_a_IJ += reduce(np.dot, (orbva, dvv_a_IJ, orbva.T))

        dmzoo_b_IJ += reduce(np.dot, (orbvb, dvv_b_IJ, orbvb.T))
    elif td_grad.base.extype == 1:
        x_ba_I, y_ab_I = x_y_I
        x_ba_J, y_ab_J = x_y_J
        if not isinstance(y_ab_I, np.ndarray):
            y_ab_I = np.zeros((noccb, nvira))
        if not isinstance(y_ab_J, np.ndarray):
            y_ab_J = np.zeros((noccb, nvira))
        x_ab_I = (-y_ab_I).T
        x_ba_I = x_ba_I.T
        x_ab_J = y_ab_J.T
        x_ba_J = x_ba_J.T
        dvv_a_IJ = np.einsum('ai,bi->ab', x_ab_I, x_ab_J)

        dvv_b_IJ = np.einsum('ai,bi->ab', x_ba_I, x_ba_J)

        doo_b_IJ = np.einsum('ai,aj->ij', x_ab_I, x_ab_J)

        doo_a_IJ = np.einsum('ai,aj->ij', x_ba_I, x_ba_J)

        dmzoo_a_IJ = reduce(np.dot, (orboa, doo_a_IJ, orboa.T))

        dmzoo_b_IJ = reduce(np.dot, (orbob, doo_b_IJ, orbob.T))

        dmzoo_a_IJ += reduce(np.dot, (orbva, dvv_a_IJ, orbva.T))

        dmzoo_b_IJ += reduce(np.dot, (orbvb, dvv_b_IJ, orbvb.T))

    mf_grad = td_grad.base._scf.nuc_grad_method()
    s1 = mf_grad.get_ovlp(mol)
    if atmlst is None:
        atmlst = range(mol.natm)
    offsetdic = mol.offset_nr_by_atom()
    nac_csf = np.zeros((len(atmlst), 3))
    for k, ia in enumerate(atmlst):
        shl0, shl1, p0, p1 = offsetdic[ia]
        nac_csf[k] -= np.einsum('xpq,pq->x', s1[:, p0:p1], dmzoo_a_IJ[p0:p1]) * 0.5
        nac_csf[k] -= np.einsum('xpq,pq->x', s1[:, p0:p1], dmzoo_b_IJ[p0:p1]) * 0.5
        nac_csf[k] += np.einsum('xqp,pq->x', s1[:, p0:p1], dmzoo_a_IJ[:, p0:p1]) * 0.5
        nac_csf[k] += np.einsum('xqp,pq->x', s1[:, p0:p1], dmzoo_b_IJ[:, p0:p1]) * 0.5
    return nac_csf


def as_scanner(NACClass):
    class Scanner(NACClass):
        def __call__(self, mol):
            td_obj = self.base
            mf_obj = td_obj._scf
            if np.allclose(mf_obj.mol.atom_coords(), mol.atom_coords()):
                return self.kernel()

            logger.info(self, 'New geometry detected. Recalculating SCF and TDDFT.')
            mf_obj.reset(mol)
            mf_obj.kernel()
            td_obj.reset(mol)
            td_obj.kernel()
            return self.kernel()

    return Scanner


def _dot_amplitude_block(a, b):
    if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
        return np.dot(a.ravel(), b.ravel())
    return 0.0


def _scale_amplitude_block(a, factor):
    if isinstance(a, np.ndarray):
        return a * factor
    return a


class NonAdiabaticCouplings(tdrhf_grad.Gradients):
    cphf_max_cycle = tdrhf_grad.Gradients.cphf_max_cycle + 20

    def __init__(self, td):
        super().__init__(td)
        self.state_I = None
        self.state_J = None
        self.ediff = False
        self.use_etfs = False
        self.x_y_I_prev = None
        self.x_y_J_prev = None
        self._keys = self._keys.union({'state_I', 'state_J', 'ediff', 'use_etfs'})

    def dump_flags(self, verbose=None):
        super().dump_flags(verbose)
        log = logger.new_logger(self, verbose)
        log.info('State I = %s', self.state_I)
        log.info('State J = %s', self.state_J)
        log.info('ediff = %s', self.ediff)
        log.info('use_etfs = %s', self.use_etfs)
        return self

    def _vector_dot(self, vec1, vec2):
        x1, y1 = vec1
        x2, y2 = vec2
        dot_x = _dot_amplitude_block(x1, x2)
        dot_y = _dot_amplitude_block(y1, y2)
        return dot_x - dot_y

    def _scale_vector(self, vec, factor):
        x, y = vec
        return (_scale_amplitude_block(x, factor), _scale_amplitude_block(y, factor))

    def compute_nac(self, state_I=None, state_J=None, atmlst=None, ediff=None, use_etfs=None, use_cache=True):
        state_I = state_I if state_I is not None else self.state_I
        state_J = state_J if state_J is not None else self.state_J
        ediff = self.ediff if ediff is None else ediff
        use_etfs = self.use_etfs if use_etfs is None else use_etfs

        if state_I is None or state_J is None:
            raise RuntimeError('state_I and state_J must be specified')

        raw_x_y_I = self.base.xy[state_I - 1]
        raw_x_y_J = self.base.xy[state_J - 1]
        e_I = self.base.e[state_I - 1]
        e_J = self.base.e[state_J - 1]

        if self.x_y_I_prev is not None and self._vector_dot(raw_x_y_I, self.x_y_I_prev) < 0:
            logger.debug(self, 'Flipping sign of state I due to phase change.')
            raw_x_y_I = self._scale_vector(raw_x_y_I, -1.0)
        self.x_y_I_prev = copy.deepcopy(raw_x_y_I)

        if self.x_y_J_prev is not None and self._vector_dot(raw_x_y_J, self.x_y_J_prev) < 0:
            logger.debug(self, 'Flipping sign of state J due to phase change.')
            raw_x_y_J = self._scale_vector(raw_x_y_J, -1.0)
        self.x_y_J_prev = copy.deepcopy(raw_x_y_J)

        hf_term = get_Hellmann_Feymann(self, raw_x_y_I, raw_x_y_J, atmlst=atmlst, state_I=state_I, state_J=state_J)
        if use_etfs:
            nac = hf_term
        else:
            nac = hf_term + nac_csf(self, raw_x_y_I, raw_x_y_J, atmlst) * (e_J - e_I)

        if ediff:
            delta_e = e_J - e_I
            if abs(delta_e) < 1e-10:
                logger.warn(self, 'Energy difference is very small: %s. NAC not divided.', delta_e)
            else:
                nac = nac / delta_e
        return nac

    def kernel(self, state_I=None, state_J=None, atmlst=None, ediff=None, use_etfs=None, use_cache=True):
        if state_I is not None:
            self.state_I = state_I
        if state_J is not None:
            self.state_J = state_J
        if atmlst is not None:
            self.atmlst = atmlst
        if ediff is not None:
            self.ediff = ediff
        if use_etfs is not None:
            self.use_etfs = use_etfs

        self.nac = self.compute_nac(
            state_I=self.state_I,
            state_J=self.state_J,
            atmlst=getattr(self, 'atmlst', None),
            ediff=self.ediff,
            use_etfs=self.use_etfs,
            use_cache=use_cache,
        )
        return self.nac

    def reset_phase(self):
        self.x_y_I_prev = None
        self.x_y_J_prev = None
        return self

    as_scanner = as_scanner


NAC = NonAdiabaticCouplings

from nest.sftda import uhf_sf

uhf_sf.TDA_SF.NAC = uhf_sf.TDDFT_SF.NAC = lib.class_as_method(NonAdiabaticCouplings)
uhf_sf.TDA_SF.nac_method = uhf_sf.TDDFT_SF.nac_method = uhf_sf.TDA_SF.NAC
