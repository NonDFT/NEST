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

"""Explicit NT-TDA matrix elements after reference-kernel reconstruction."""

import numpy as np
from pyscf import ao2mo, dft, lib
from pyscf.lib import logger


def get_ab(mf, deltaS=-1, nobeta=False):
    r"""Return the dense A matrix for a ROKS reference (there is no B matrix).

    i,j denote closed-shell orbitals; a,b virtual orbitals; u,v,t,w open-shell
    orbitals. Amplitudes store the hole before the particle: X[i,a] for CV,
    X[i,u] for CO, X[u,a] for OV, and X[u,t] for the OO amplitude in A_{tu,vw}.

    deltaS=-1 uses [[CO, CV], [OO, OV]].ravel(), including its redundant
    OO identity zero mode. deltaS=0 concatenates CO, CV, scalar OO, OV, CV0.
    deltaS=+1 uses CV.ravel(). For a batch x of flattened row vectors,
    vind(x) = (A @ x.T).T.

    K[p,q,r,s] is K^Ref_{pq,rs}, and Fz[p,q] = 1/2 sum_u K[p,q,u,u].
    XC kernels are contracted directly with MO values and derivatives;
    neither response functions nor gen_vind are used in this construction.
    This dense implementation stores full four-index MO tensors and is
    intended for small systems. With density fitting, both J and K must use
    the same fitted integrals (only_dfj=True is not supported).
    """
    if not isinstance(mf, (dft.roks.ROKS, dft.rks_symm.SymAdaptedROKS)):
        raise TypeError('NTTDA get_ab requires a ROKS reference')
    if deltaS not in (-1, 0, 1):
        raise ValueError('deltaS must be -1, 0, or 1')
    with_df = getattr(mf, 'with_df', None)
    if with_df and getattr(mf, 'only_dfj', False):
        raise NotImplementedError('NTTDA get_ab does not support only_dfj=True')

    mol = mf.mol
    csidx = np.where(mf.mo_occ == 2)[0]
    osidx = np.where(mf.mo_occ == 1)[0]
    vsidx = np.where(mf.mo_occ == 0)[0]
    nc, no, nv = len(csidx), len(osidx), len(vsidx)
    s = no * 0.5
    if deltaS == -1:
        assert s >= 1, 'NTTDA for Sf=Si-1 only supports case that Si>=1.'
    elif deltaS == 0:
        assert s >= 0.5, 'NTTDA only supports case that Sf=Si>=1/2.'
    assert s == mol.spin * 0.5
    order = np.concatenate((csidx, osidx, vsidx))
    mo = mf.mo_coeff[:, order]
    assert mo.dtype == np.double
    nmo = nc + no + nv
    c, o, v = slice(0, nc), slice(nc, nc + no), slice(nc + no, nmo)
    ic, io, iv = np.eye(nc), np.eye(no), np.eye(nv)

    ni = mf._numint
    ni.libxc.test_deriv_order(mf.xc, 2, raise_error=True)
    xctype = ni._xc_type(mf.xc)
    if xctype not in ('HF', 'LDA', 'GGA', 'MGGA'):
        raise NotImplementedError('NTTDA get_ab does not support XC type %s' % xctype)
    if mf.do_nlc():
        logger.warn(mf, 'NLC contribution in get_ab is NOT included')
    omega, alpha, hyb = ni.rsh_and_hybrid_coeff(mf.xc, mol.spin)

    # Reference kernel: exact exchange K_{pq,rs} = -(pr|qs).
    kref = np.zeros((nmo, nmo, nmo, nmo))
    if hyb != 0:
        if with_df:
            eri = with_df.ao2mo(mo, compact=False)
        else:
            eri = ao2mo.kernel(mol, mo, compact=False)
        eri = eri.reshape(nmo, nmo, nmo, nmo)
        kref -= hyb * lib.einsum('prqs->pqrs', eri)
    if omega != 0:
        if with_df:
            with with_df.range_coulomb(omega) as rsh_df:
                eri = rsh_df.ao2mo(mo, compact=False)
        else:
            with mol.with_range_coulomb(omega):
                eri = ao2mo.kernel(mol, mo, compact=False)
        eri = eri.reshape(nmo, nmo, nmo, nmo)
        kref -= (alpha - hyb) * lib.einsum('prqs->pqrs', eri)

    # The reference XC Hessian is evaluated at rho_alpha = rho_beta = rho/2.
    # Pair features are phi_p phi_q, grad(phi_p phi_q), and (for MGGA)
    # 1/2 grad(phi_p).grad(phi_q). Contract both pairs with fxc and grid weights.
    if xctype != 'HF':
        if mf.grids.coords is None:
            mf.grids.build(with_non0tab=True)
        max_memory = max(2000, mf.max_memory * .8 - lib.current_memory()[0])
        ao_deriv = 0 if xctype == 'LDA' else 1
        for ao, mask, weight, coords in ni.block_loop(mol, mf.grids, mol.nao_nr(), ao_deriv, max_memory):
            rho = ni.eval_rho2(mol, ao, mo, mf.mo_occ[order], mask, xctype, with_lapl=False) * .5
            fxc = ni.eval_xc_eff(mf.xc, (rho, rho), deriv=2, xctype=xctype, spin=1)[2]
            fxc_ref = .5 * (fxc[0, :, 0] - fxc[0, :, 1] - fxc[1, :, 0] + fxc[1, :, 1])
            if xctype == 'LDA':
                phi = lib.einsum('gp,pi->gi', ao, mo)
                pair = lib.einsum('gp,gq->gpq', phi, phi)
                weighted_pair = pair * (fxc_ref[0, 0] * weight)[:, None, None]
                kref += lib.einsum('gpq,grs->pqrs', pair, weighted_pair)
            else:
                phi = lib.einsum('xgp,pi->xgi', ao, mo)
                pair = lib.einsum('xgp,gq->xgpq', phi, phi[0])
                pair[1:4] += lib.einsum('gp,xgq->xgpq', phi[0], phi[1:4])
                if xctype == 'MGGA':
                    tau_pair = .5 * lib.einsum('xgp,xgq->gpq', phi[1:4], phi[1:4])
                    pair = np.concatenate((pair, tau_pair[None]), axis=0)
                weighted_pair = lib.einsum('xyg,ygpq->xgpq', fxc_ref * weight, pair)
                kref += lib.einsum('xgpq,xgrs->pqrs', pair, weighted_pair)

    # Reconstructed Fz and spin-averaged F0, both in the (C, O, V) MO basis.
    fz = .5 * lib.einsum('pquu->pq', kref[:, :, o, o])
    if nobeta:
        dma, dmb = mf.make_rdm1()
        dm0 = .5 * (dma + dmb)
        fock = mf.get_fock(dm=np.array([dm0, dm0]))
    else:
        fock = mf.get_fock()
    f0 = mo.T @ (.5 * (fock.focka + fock.fockb)) @ mo
    fp, fm = f0 + fz, f0 - fz

    if deltaS == 1:
        # CV-CV: delta_ij (F0+Fz)_ab - delta_ab (F0-Fz)_ji + K_ai,bj.
        a = lib.einsum('ij,ab->iajb', ic, fp[v, v])
        a -= lib.einsum('ab,ji->iajb', iv, fm[c, c])
        a += lib.einsum('aibj->iajb', kref[v, c, v, c])
        return a.reshape(nc * nv, nc * nv)

    if deltaS == -1:
        # CV-CV
        a_cvcv = lib.einsum('ij,ab->iajb', ic, fm[v, v])
        a_cvcv -= lib.einsum('ab,ji->iajb', iv, fp[c, c])
        a_cvcv -= lib.einsum('ij,ab->iajb', ic, fz[v, v]) / s
        a_cvcv -= lib.einsum('ab,ji->iajb', iv, fz[c, c]) / s
        a_cvcv += lib.einsum('aibj->iajb', kref[v, c, v, c])

        # CV-CO and CV-OV
        a_cvco = lib.einsum('ij,av->iajv', ic, fm[v, o])
        a_cvco += lib.einsum('aivj->iajv', kref[v, c, o, c])
        a_cvco *= np.sqrt((2 * s + 1) / (2 * s))
        a_cvov = -lib.einsum('ab,vi->iavb', iv, fp[o, c])
        a_cvov += lib.einsum('aibv->iavb', kref[v, c, v, o])
        a_cvov *= np.sqrt((2 * s + 1) / (2 * s))

        # CO-CO
        a_coco = lib.einsum('ij,uv->iujv', ic, fm[o, o])
        a_coco -= lib.einsum('uv,ji->iujv', io, fp[c, c])
        a_coco -= 2 / (2 * s - 1) * lib.einsum('uv,ji->iujv', io, fz[c, c])
        a_coco += lib.einsum('uivj->iujv', kref[o, c, o, c])
        a_coco += lib.einsum('uvij->iujv', kref[o, o, c, c]) / (2 * s - 1)

        # CO-OV
        a_coov = -lib.einsum('ubiv->iuvb', kref[o, v, c, o]) / (2 * s - 1)
        a_coov += 2 * s / (2 * s - 1) * lib.einsum('uibv->iuvb', kref[o, c, v, o])

        # OV-OV
        a_ovov = lib.einsum('uv,ab->uavb', io, fm[v, v])
        a_ovov -= lib.einsum('ab,vu->uavb', iv, fp[o, o])
        a_ovov -= 2 / (2 * s - 1) * lib.einsum('uv,ab->uavb', io, fz[v, v])
        a_ovov += lib.einsum('aubv->uavb', kref[v, o, v, o])
        a_ovov += lib.einsum('abuv->uavb', kref[v, v, o, o]) / (2 * s - 1)

        # OO-CV: row indices are (u,t), corresponding to A_tu,bj.
        a_oocv = -lib.einsum('ut,jb->utjb', io, fz[c, v]) / s
        a_oocv += lib.einsum('tubj->utjb', kref[o, o, v, c])
        a_oocv *= np.sqrt((2 * s + 1) / (2 * s - 1))

        # OO-CO
        a_ooco = -np.sqrt(2 * s / (2 * s - 1)) * lib.einsum('vt,ju->utjv', io, fp[c, o])
        a_ooco += lib.einsum('ut,jv->utjv', io, fm[c, o]) / np.sqrt(2 * s * (2 * s - 1))
        a_ooco += np.sqrt(2 * s / (2 * s - 1)) * lib.einsum('tuvj->utjv', kref[o, o, o, c])

        # OO-OV
        a_ooov = np.sqrt(2 * s / (2 * s - 1)) * lib.einsum('uv,tb->utvb', io, fm[o, v])
        a_ooov -= lib.einsum('tu,vb->utvb', io, fp[o, v]) / np.sqrt(2 * s * (2 * s - 1))
        a_ooov += np.sqrt(2 * s / (2 * s - 1)) * lib.einsum('tubv->utvb', kref[o, o, v, o])

        # OO-OO
        a_oooo = lib.einsum('wu,tv->utwv', io, fm[o, o])
        a_oooo -= lib.einsum('tv,wu->utwv', io, fp[o, o])
        a_oooo += lib.einsum('tuvw->utwv', kref[o, o, o, o])

        # Native [[CO, CV], [OO, OV]] layout; transpose the reverse blocks.
        nocc, nvir = nc + no, no + nv
        a = np.empty((nocc, nvir, nocc, nvir))
        a[:nc, :no, :nc, :no] = a_coco
        a[:nc, :no, :nc, no:] = a_cvco.transpose(2, 3, 0, 1)
        a[:nc, :no, nc:, :no] = a_ooco.transpose(2, 3, 0, 1)
        a[:nc, :no, nc:, no:] = a_coov
        a[:nc, no:, :nc, :no] = a_cvco
        a[:nc, no:, :nc, no:] = a_cvcv
        a[:nc, no:, nc:, :no] = a_oocv.transpose(2, 3, 0, 1)
        a[:nc, no:, nc:, no:] = a_cvov
        a[nc:, :no, :nc, :no] = a_ooco
        a[nc:, :no, :nc, no:] = a_oocv
        a[nc:, :no, nc:, :no] = a_oooo
        a[nc:, :no, nc:, no:] = a_ooov
        a[nc:, no:, :nc, :no] = a_coov.transpose(2, 3, 0, 1)
        a[nc:, no:, :nc, no:] = a_cvov.transpose(2, 3, 0, 1)
        a[nc:, no:, nc:, :no] = a_ooov.transpose(2, 3, 0, 1)
        a[nc:, no:, nc:, no:] = a_ovov
        return a.reshape(nocc * nvir, nocc * nvir)

    # deltaS=0: CV0-CV0
    a_cv0cv0 = lib.einsum('ij,ab->iajb', ic, f0[v, v])
    a_cv0cv0 -= lib.einsum('ab,ji->iajb', iv, f0[c, c])
    a_cv0cv0 += lib.einsum('aibj->iajb', kref[v, c, v, c])
    a_cv0cv0 -= 2 * lib.einsum('abij->iajb', kref[v, v, c, c])

    # CV0-CV
    a_cv0cv = -lib.einsum('ij,ab->iajb', ic, fz[v, v])
    a_cv0cv += lib.einsum('ab,ji->iajb', iv, fz[c, c])
    a_cv0cv *= np.sqrt((s + 1) / s)

    # CV0-CO
    a_cv0co = lib.einsum('ij,av->iajv', ic, fm[v, o]) / np.sqrt(2)
    a_cv0co += lib.einsum('aivj->iajv', kref[v, c, o, c]) / np.sqrt(2)
    a_cv0co -= np.sqrt(2) * lib.einsum('avij->iajv', kref[v, o, c, c])

    # CV0-OV
    a_cv0ov = lib.einsum('ab,vi->iavb', iv, fp[o, c]) / np.sqrt(2)
    a_cv0ov -= lib.einsum('aibv->iavb', kref[v, c, v, o]) / np.sqrt(2)
    a_cv0ov += np.sqrt(2) * lib.einsum('abiv->iavb', kref[v, v, c, o])

    # CV-CV
    a_cvcv = lib.einsum('ij,ab->iajb', ic, f0[v, v] - fz[v, v] / s)
    a_cvcv -= lib.einsum('ab,ji->iajb', iv, f0[c, c] + fz[c, c] / s)
    a_cvcv += lib.einsum('aibj->iajb', kref[v, c, v, c])

    # CV-CO and CV-OV
    a_cvco = lib.einsum('ij,av->iajv', ic, fm[v, o])
    a_cvco += lib.einsum('aivj->iajv', kref[v, c, o, c])
    a_cvco *= np.sqrt((s + 1) / (2 * s))
    a_cvov = -lib.einsum('ab,vi->iavb', iv, fp[o, c])
    a_cvov += lib.einsum('aibv->iavb', kref[v, c, v, o])
    a_cvov *= np.sqrt((s + 1) / (2 * s))

    # CO-CO and CO-OV
    a_coco = lib.einsum('ij,uv->iujv', ic, fm[o, o])
    a_coco -= lib.einsum('uv,ji->iujv', io, fm[c, c])
    a_coco += lib.einsum('uivj->iujv', kref[o, c, o, c])
    a_coco -= lib.einsum('uvij->iujv', kref[o, o, c, c])
    a_coov = lib.einsum('ubiv->iuvb', kref[o, v, c, o])

    # OV-OV
    a_ovov = lib.einsum('uv,ab->uavb', io, fp[v, v])
    a_ovov -= lib.einsum('ab,vu->uavb', iv, fp[o, o])
    a_ovov += lib.einsum('aubv->uavb', kref[v, o, v, o])
    a_ovov -= lib.einsum('abuv->uavb', kref[v, v, o, o])

    # The OO block is a scalar in this sector.
    a_oocv0 = -np.sqrt(2) * f0[c, v].reshape(1, -1)
    a_oocv = 2 * np.sqrt((s + 1) / (2 * s)) * fz[c, v].reshape(1, -1)
    a_ooco = -fm[c, o].reshape(1, -1)
    a_ooov = fp[o, v].reshape(1, -1)

    # Native order: CO, CV, scalar OO, OV, CV0.
    nco, ncv, nov = nc * no, nc * nv, no * nv
    a_coco = a_coco.reshape(nco, nco)
    a_cvcv = a_cvcv.reshape(ncv, ncv)
    a_ovov = a_ovov.reshape(nov, nov)
    a_cvco = a_cvco.reshape(ncv, nco)
    a_cvov = a_cvov.reshape(ncv, nov)
    a_coov = a_coov.reshape(nco, nov)
    a_cv0cv0 = a_cv0cv0.reshape(ncv, ncv)
    a_cv0cv = a_cv0cv.reshape(ncv, ncv)
    a_cv0co = a_cv0co.reshape(ncv, nco)
    a_cv0ov = a_cv0ov.reshape(ncv, nov)
    return np.block([
        [a_coco,   a_cvco.T, a_ooco.T,         a_coov,   a_cv0co.T],
        [a_cvco,   a_cvcv,   a_oocv.T,         a_cvov,   a_cv0cv.T],
        [a_ooco,   a_oocv,   np.zeros((1, 1)), a_ooov,   a_oocv0],
        [a_coov.T, a_cvov.T, a_ooov.T,         a_ovov,   a_cv0ov.T],
        [a_cv0co,  a_cv0cv,  a_oocv0.T,        a_cv0ov,  a_cv0cv0],
    ])
