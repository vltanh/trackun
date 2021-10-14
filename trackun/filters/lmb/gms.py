from dataclasses import dataclass
from typing import List, Tuple

from trackun.filters.base import GMSFilter
from trackun.common.gaussian_mixture import GaussianMixture
from trackun.common.kalman_filter import KalmanFilter
from trackun.common.gating import EllipsoidallGating
from trackun.common.kshortest import find_k_shortest_path
from trackun.common.murty import find_m_best_assignment

import numpy as np

__all__ = [
    'LMB_GMS_Data',
    'LMB_GMS_Filter',
]

EPS = np.finfo(np.float64).eps
INF = np.inf


@dataclass
class Track:
    gm: GaussianMixture
    l: Tuple[int, int]
    ah: List[int]


@dataclass
class LMB_GMS_Data:
    track_table: List[Track]
    r: np.ndarray
    w: np.ndarray
    I: List[np.ndarray]
    n: np.ndarray
    cdn: np.ndarray


@dataclass
class LMB:
    track_table: List[Track]
    r: np.ndarray


@dataclass
class GLMB:
    track_table: List[Track]
    w: np.ndarray
    I: List[np.ndarray]
    n: np.ndarray
    cdn: np.ndarray


def logsumexp(w):
    val = np.max(w)
    return np.log(np.sum(np.exp(w - val))) + val


def esf(Z):
    n_z = len(Z)
    if n_z == 0:
        return np.array([1.])

    F = np.zeros((2, n_z + 1))
    F[0, 1] = Z[0]

    i_n, i_nminus = 1, 0
    for n in range(2, n_z + 1):
        F[i_n, 1] = F[i_nminus, 1] + Z[n - 1]
        for k in range(2, n):
            F[i_n, k] = F[i_nminus, k] + Z[n - 1] * F[i_nminus, k - 1]
        F[i_n, n] = Z[n - 1] * F[i_nminus, n - 1]
        i_n, i_nminus = i_nminus, i_n

    F[i_nminus, 0] = 1
    return F[i_nminus]


def mbest_wrap(P, m):
    n1, n2 = P.shape

    # Padding
    P0 = np.zeros((n1, n2 + n1))
    P0[:, :n2] = P

    # Murty
    assignments, costs = find_m_best_assignment(P0, m)
    costs = costs.reshape(-1)

    # Remove padding
    assignments = assignments[:, :n1]

    # Set clututer
    assignments[assignments >= n2] = -1

    return assignments, costs


def kshortest_wrap(rs, k):
    if k == 0:
        return [], []

    ns = len(rs)
    _is = np.argsort(-rs)
    ds = rs[_is]

    CM = np.full((ns, ns), INF)
    for i in range(ns):
        CM[:i, i] = ds[i]

    CMPad = np.full((ns + 2, ns + 2), INF)
    CMPad[0, 1:-1] = ds
    CMPad[0, -1] = 0.
    CMPad[1:-1, -1] = 0.
    CMPad[1:-1, 1:-1] = CM

    paths, costs = find_k_shortest_path(CMPad, 0, ns + 1, k)

    for p in range(len(paths)):
        if np.array_equal(paths[p], np.array([1, ns + 2])):
            paths[p] = []
        else:
            paths[p] = [x - 1 for x in paths[p][1:-1]]
            paths[p] = _is[paths[p]]
    return paths, costs


def select(upd, idxkeep):
    tt = [upd.track_table[i] for i in idxkeep]
    r = upd.r[idxkeep]
    return LMB(tt, r)


def prune(upd: LMB,
          T_threshold: float) -> LMB:
    idxkeep = np.where(upd.r > T_threshold)[0]
    return select(upd, idxkeep)


def cap(upd: LMB,
        T_max: int) -> LMB:
    if len(upd.r) > T_max:
        idxkeep = upd.r.argsort()[::-1][:T_max]
        return select(upd, idxkeep)
    else:
        return upd


def get_rvals(tt):
    rvect = np.empty(len(tt))
    for tabidx in range(len(tt)):
        rvect[tabidx] = tt[tabidx].r
    return rvect


def lmb2glmb(rvect: np.ndarray, tt: List[Track], H: int):
    costv = rvect / (1 - rvect)
    neglogcostv = -np.log(costv)
    bpaths, nlcost = kshortest_wrap(neglogcostv, H)

    w = np.empty(len(nlcost))
    I = []
    n = np.empty(len(nlcost), dtype=np.int64)
    for hidx in range(len(nlcost)):
        w[hidx] = np.log(1 - rvect).sum() - nlcost[hidx]
        I.append(bpaths[hidx])
        n[hidx] = len(bpaths[hidx])
    w = np.exp(w - logsumexp(w))

    return w, I, n


class LMB_GMS_Filter(GMSFilter):
    def __init__(self,
                 model,
                 L_max=100,
                 elim_thres=1e-5,
                 merge_threshold=4,
                 use_gating=True,
                 pG=0.9999999,
                 T_max=100,
                 track_threshold=1e-3,
                 H_bth=5,
                 H_sur=3000,
                 H_upd=3000,
                 H_max=3000,
                 hyp_thres=1e-15) -> None:
        super().__init__(model,
                         L_max, elim_thres, merge_threshold,
                         use_gating, pG)
        self.T_max = T_max
        self.track_threshold = track_threshold

        self.H_bth = H_bth
        self.H_sur = H_sur
        self.H_upd = H_upd
        self.H_max = H_max
        self.hyp_thres = hyp_thres

    def init(self) -> LMB_GMS_Data:
        super().init()
        track_table = []
        r = np.empty(0)
        w = np.ones(1)
        I = [np.empty(0, dtype=np.int64)]
        n = np.zeros(1).astype(np.int64)
        cdn = np.ones(1)
        return LMB_GMS_Data(track_table, r, w, I, n, cdn)

    def predict(self, upd):
        # === Birth ===

        # Create birth tracks
        r_birth = self.model.birth_model.rs
        tt_birth = []
        for tabbidx in range(len(self.model.birth_model.rs)):
            gm = self.model.birth_model.gms[tabbidx]
            l = (self.k, tabbidx)
            ah = []
            tt_birth.append(Track(gm, l, ah))
        w_birth, I_birth, n_birth = lmb2glmb(r_birth, tt_birth, self.H_bth)

        # === Survive ===

        pS = self.model.survival_model.get_probability()

        # Create surviving tracks
        r_surv = pS * upd.r
        tt_surv = []
        for tabsidx in range(len(upd.track_table)):
            mtemp_predict, Ptemp_predict = KalmanFilter.predict(self.model.motion_model.F,
                                                                self.model.motion_model.Q,
                                                                upd.track_table[tabsidx].gm.m,
                                                                upd.track_table[tabsidx].gm.P)

            w_surv = upd.track_table[tabsidx].gm.w
            m_surv = mtemp_predict
            P_surv = Ptemp_predict

            gm_surv = GaussianMixture(w_surv, m_surv, P_surv)
            l_surv = upd.track_table[tabsidx].l
            ah_surv = upd.track_table[tabsidx].ah

            tt_surv.append(Track(gm_surv, l_surv, ah_surv))
        w_surv, I_surv, n_surv = lmb2glmb(r_surv, tt_surv, self.H_sur)

        # Combine birth and surviving tracks
        tt_pred = tt_birth + tt_surv

        # r_pred = np.hstack([r_birth, r_surv])

        w_pred = (w_birth[:, np.newaxis] * w_surv[np.newaxis, :]).reshape(-1)
        w_pred = w_pred / w_pred.sum()

        I_pred = [
            np.hstack([I_birth[bidx], len(tt_birth) + I_surv[sidx]])
            for bidx in range(len(w_birth))
            for sidx in range(len(w_surv))
        ]

        n_pred = (n_birth[:, np.newaxis] + n_surv[np.newaxis, :]).reshape(-1)

        # cdn_pred = np.zeros(n_pred.max() + 1)
        # for card in range(cdn_pred.shape[0]):
        #     cdn_pred[card] = np.sum(w_pred[n_pred == card])

        # w_pred, I_pred, n_pred = clean_predict(w_pred, I_pred, n_pred)
        pred = LMB_GMS_Data(tt_pred, None, w_pred, I_pred, n_pred, None)
        # print(pred)
        # input()

        return pred

    def gating(self,
               Z: np.ndarray,
               pred: LMB_GMS_Data) -> np.ndarray:
        m_tracks = []
        P_tracks = []
        for tabidx in range(len(pred.track_table)):
            m_tracks.append(pred.track_table[tabidx].gm.m)
            P_tracks.append(pred.track_table[tabidx].gm.P)

        m_tracks = np.vstack(m_tracks)
        P_tracks = np.vstack(P_tracks)
        return EllipsoidallGating.filter(Z,
                                         self.gamma,
                                         self.model.measurement_model.H,
                                         self.model.measurement_model.R,
                                         m_tracks, P_tracks)

    def update(self,
               Z: np.ndarray,
               pred: LMB_GMS_Data) -> LMB_GMS_Data:
        # == Gating ==
        cand_Z = self.gating(Z, pred) \
            if self.use_gating \
            else Z

        # == Update ==

        # Create update tracks
        N1 = len(pred.track_table)
        N2 = cand_Z.shape[0]
        M = N1 * (N2 + 1)
        tt_upda = [None for _ in range(M)]

        # Missed detection tracks
        for tabidx in range(N1):
            gm = pred.track_table[tabidx].gm
            l = pred.track_table[tabidx].l
            ah = pred.track_table[tabidx].ah + [0]
            tt_upda[tabidx] = Track(gm, l, ah)

        # Updated tracks
        allcostm = np.empty((N1, N2))
        for tabidx in range(N1):
            qz_temp, m_temp, P_temp = KalmanFilter.update(cand_Z,
                                                          self.model.measurement_model.H,
                                                          self.model.measurement_model.R,
                                                          pred.track_table[tabidx].gm.m,
                                                          pred.track_table[tabidx].gm.P)
            for emm in range(N2):
                w_temp = qz_temp[:, emm] * pred.track_table[tabidx].gm.w + EPS

                gm = GaussianMixture(w_temp / w_temp.sum(),
                                     m_temp[:, emm], P_temp)
                l = pred.track_table[tabidx].l
                ah = pred.track_table[tabidx].ah + [emm + 1]

                stoidx = N1 * (emm + 1) + (tabidx + 1) - 1
                tt_upda[stoidx] = Track(gm, l, ah)

                allcostm[tabidx, emm] = w_temp.sum()

        # Component update
        lambda_c = self.model.clutter_model.lambda_c
        pdf_c = self.model.clutter_model.pdf_c
        pD = self.model.detection_model.get_probability()

        if N2 == 0:
            w_upda = -lambda_c + pred.n * np.log1p(-pD) + np.log(pred.w)
            I_upda = pred.I
            n_upda = pred.n
        else:
            w_upda, I_upda, n_upda = [], [], []
            for pidx in range(len(pred.w)):
                w = -lambda_c + N2 * np.log(lambda_c * pdf_c)
                if pred.n[pidx] == 0:
                    w_p = w + np.log(pred.w[pidx])
                    I_p = pred.I[pidx]
                    n_p = pred.n[pidx]

                    w_upda.append(w_p)
                    I_upda.append(I_p)
                    n_upda.append(n_p)
                else:
                    costm = pD / (1 - pD) * \
                        allcostm[pred.I[pidx]] / (lambda_c * pdf_c)
                    neglogcostm = -np.log(costm)

                    N = int(self.H_upd *
                            np.sqrt(pred.w[pidx]) / np.sum(np.sqrt(pred.w))
                            + 0.5)

                    if N:
                        uasses, nlcost = mbest_wrap(neglogcostm, N)

                        w_p = w + pred.n[pidx] * np.log1p(-pD) \
                                + np.log(pred.w[pidx])
                        for hidx in range(len(nlcost)):
                            w_ph = w_p - nlcost[hidx]
                            I_ph = N1 * (uasses[hidx] + 1) + pred.I[pidx]
                            n_ph = pred.n[pidx]

                            w_upda.append(w_ph)
                            I_upda.append(I_ph)
                            n_upda.append(n_ph)
            n_upda = np.array(n_upda).astype(np.int64)
        w_upda = np.exp(w_upda - logsumexp(w_upda))
        # print(w_upda)
        # print(I_upda)
        # print(n_upda)
        # input()

        # cdn_upda = np.zeros(n_upda.max() + 1)
        # for card in range(cdn_upda.shape[0]):
        #     cdn_upda[card] = np.sum(w_upda[n_upda == card])
        # print(cdn_upda)
        # input()

        # GLMB to LMB
        lmat = np.empty((len(tt_upda), 2), dtype=np.int64)
        for tabidx in range(len(tt_upda)):
            lmat[tabidx] = tt_upda[tabidx].l
        # print(lmat)
        # input()

        cu, ia, ic = np.unique(lmat, axis=0,
                               return_index=True,
                               return_inverse=True)
        # print(cu)
        # print(ia)
        # print(ic)
        # input()

        w_upda_new = [[] for _ in range(len(cu))]
        m_upda_new = [[] for _ in range(len(cu))]
        P_upda_new = [[] for _ in range(len(cu))]
        for hidx in range(len(w_upda)):
            for t in range(n_upda[hidx]):
                trkidx = I_upda[hidx][t]
                newidx = ic[trkidx]

                w_upda_new[newidx].append(w_upda[hidx] * tt_upda[trkidx].gm.w)
                m_upda_new[newidx].append(tt_upda[trkidx].gm.m)
                P_upda_new[newidx].append(tt_upda[trkidx].gm.P)

        r_upda_new = np.zeros(len(cu))
        tt_upda_new = []
        for newidx in range(len(cu)):
            w_upda_new[newidx] = np.hstack(w_upda_new[newidx])
            r_upda_new[newidx] = w_upda_new[newidx].sum()
            w_upda_new[newidx] /= r_upda_new[newidx]

            m_upda_new[newidx] = np.vstack(m_upda_new[newidx])
            P_upda_new[newidx] = np.vstack(P_upda_new[newidx])

            gm = GaussianMixture(w_upda_new[newidx],
                                 m_upda_new[newidx],
                                 P_upda_new[newidx])
            l = cu[newidx]
            ah = tt_upda[ia[newidx]].ah

            tt_upda_new.append(Track(gm, l, ah))

        upd = LMB(tt_upda_new, r_upda_new)

        upd = prune(upd, self.track_threshold)
        upd = cap(upd, self.T_max)

        for i in range(len(upd.track_table)):
            upd.track_table[i].gm = \
                upd.track_table[i].gm.prune(self.elim_threshold)
            upd.track_table[i].gm = \
                upd.track_table[i].gm.merge_and_cap(
                    self.merge_threshold, self.L_max)

        upd = LMB_GMS_Data(upd.track_table, upd.r,
                           None, None, None, None)

        return upd

    def visualizable_estimate(self, upd: LMB_GMS_Data) -> Track:
        cdn = np.prod(1 - upd.r) * esf(upd.r / (1 - upd.r))
        M = np.argmax(cdn)
        M = min(len(upd.r), M)

        ah = [None for _ in range(M)]
        w = np.empty(M)
        X = np.empty((M, self.model.motion_model.x_dim))
        P = np.empty((M, self.model.motion_model.x_dim,
                     self.model.motion_model.x_dim))
        L = np.empty((M, 2), dtype=np.int64)

        idxcmp = upd.r.argsort()[::-1]
        for m in range(M):
            track = upd.track_table[idxcmp[m]]
            w[m], X[m], P[m] = track.gm.cap(1).unpack()
            L[m] = track.l
            ah[m] = track.ah

        return Track(GaussianMixture(w, X, P), L, ah)
