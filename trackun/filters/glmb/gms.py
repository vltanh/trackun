from dataclasses import dataclass
from typing import List

from trackun.filters.base import GMSFilter
from trackun.common.gaussian_mixture import GaussianMixture
from trackun.common.kalman_filter import KalmanFilter
from trackun.common.gating import EllipsoidallGating
from trackun.common.kshortest import kshortestwrap_pred
from trackun.common.murty import mbest_wrap

import numpy as np

__all__ = [
    'GLMB_GMS_Data',
    'GLMB_GMS_Filter',
]

EPS = np.finfo(np.float64).eps


@dataclass
class Track:
    gm: GaussianMixture
    l: List[int]
    ah: List[int]


@dataclass
class GLMB_GMS_Data:
    track_table: List[Track]
    w: np.ndarray
    I: List[List[int]]
    n: np.ndarray
    cdn: np.ndarray


def logsumexp(w):
    if np.all(w == -np.inf):
        return -np.inf
    val = np.max(w)
    return np.log(np.sum(np.exp(w - val))) + val


def clean_predict(pred: GLMB_GMS_Data) -> GLMB_GMS_Data:
    N = len(pred.w)

    _hash = [None for _ in range(N)]
    for hidx in range(N):
        _hash[hidx] = '*'.join(map(lambda x: str(x+1),
                               sorted(pred.I[hidx]))) + '*'
    # print(_hash)
    # input()

    cu, _, ic = np.unique(_hash,
                          return_index=True,
                          return_inverse=True)
    # print(cu)
    # print(ic)
    # input()

    tt_temp = pred.track_table
    w_temp = np.zeros(len(cu))
    I_temp = [None for _ in range(len(cu))]
    n_temp = np.zeros(len(cu), dtype=np.int32)
    for hidx in range(len(ic)):
        w_temp[ic[hidx]] = w_temp[ic[hidx]] + pred.w[hidx]
        I_temp[ic[hidx]] = pred.I[hidx]
        n_temp[ic[hidx]] = pred.n[hidx]
    cdn_temp = pred.cdn

    return GLMB_GMS_Data(tt_temp, w_temp, I_temp, n_temp, cdn_temp)


def clean_update(upd: GLMB_GMS_Data) -> GLMB_GMS_Data:
    usedindicator = np.zeros(len(upd.track_table), dtype=np.int32)
    for hidx in range(len(upd.w)):
        usedindicator[upd.I[hidx]] += 1
    trackcount = np.sum(usedindicator > 0)
    # print(usedindicator, trackcount)
    # input()

    newindices = np.zeros(len(upd.track_table), dtype=np.ndarray)
    newindices[usedindicator > 0] = np.arange(1, trackcount + 1)
    # print(newindices)
    # input()

    tt_temp = [x for x, y in zip(upd.track_table, usedindicator) if y > 0]
    w_temp = upd.w.copy()
    I_temp = [(newindices - 1)[upd.I[hidx]].copy().tolist()
              for hidx in range(len(upd.w))]
    # print(I_temp)
    # input()
    n_temp = upd.n.copy()
    cdn_temp = upd.cdn.copy()

    return GLMB_GMS_Data(tt_temp, w_temp, I_temp, n_temp, cdn_temp)


def select(upd, idxkeep):
    tt_prune = [Track(x.gm.copy(), [x.l[0], x.l[1]], [xx for xx in x.ah])
                for x in upd.track_table]
    w_prune = upd.w[idxkeep].copy()
    I_prune = [upd.I[x] for x in idxkeep]
    n_prune = upd.n[idxkeep].copy()

    w_prune = w_prune / w_prune.sum()

    N = n_prune.max() + 1
    cdn_prune = np.zeros(N)
    for card in range(N):
        cdn_prune[card] = w_prune[n_prune == card].sum()

    return GLMB_GMS_Data(tt_prune, w_prune, I_prune, n_prune, cdn_prune)


def prune(upd: GLMB_GMS_Data,
          hyp_threshold: float) -> GLMB_GMS_Data:
    idxkeep = np.where(upd.w > hyp_threshold)[0]
    return select(upd, idxkeep)


def cap(upd: GLMB_GMS_Data,
        H_max: int) -> GLMB_GMS_Data:
    if len(upd.w) > H_max:
        idxkeep = upd.w.argsort()[::-1][:H_max]
        return select(upd, idxkeep)
    else:
        return upd


class GLMB_GMS_Filter(GMSFilter):
    def __init__(self,
                 model,
                 L_max=100,
                 elim_thres=1e-5,
                 merge_threshold=4,
                 use_gating=True,
                 pG=0.9999999,
                 H_bth=5,
                 H_sur=3000,
                 H_upd=3000,
                 H_max=3000,
                 hyp_thres=1e-15) -> None:
        super().__init__(model,
                         L_max, elim_thres, merge_threshold,
                         use_gating, pG)
        self.H_bth = H_bth
        self.H_sur = H_sur
        self.H_upd = H_upd
        self.H_max = H_max
        self.hyp_thres = hyp_thres

    def init(self) -> GLMB_GMS_Data:
        super().init()
        track_table = []
        w = np.ones(1)
        I = [[]]
        n = np.zeros(1).astype(np.int32)
        cdn = np.ones(1)
        return GLMB_GMS_Data(track_table, w, I, n, cdn)

    def predict(self,
                upd: GLMB_GMS_Data) -> GLMB_GMS_Data:
        # === Birth ===

        # Create birth tracks
        N_birth = len(self.model.birth_model.rs)
        tt_birth = [None for _ in range(N_birth)]
        for tabbidx in range(N_birth):
            gm = self.model.birth_model.gms[tabbidx].copy()
            l = [self.k, tabbidx]
            ah = []
            tt_birth[tabbidx] = Track(gm, l, ah)

        # Calculate best birth hypotheses
        r_birth = self.model.birth_model.rs
        costv = r_birth / (1 - r_birth)
        neglogcostv = -np.log(costv)
        bpaths, nlcost = kshortestwrap_pred(neglogcostv, self.H_bth)

        # Generate corresponding birth hypotheses
        w_birth = np.zeros(len(nlcost))
        I_birth = [None for _ in range(len(nlcost))]
        n_birth = np.zeros(len(nlcost), dtype=np.int32)
        for hidx in range(len(nlcost)):
            w_birth[hidx] = np.log(1 - r_birth).sum() - nlcost[hidx]
            I_birth[hidx] = bpaths[hidx]
            n_birth[hidx] = len(bpaths[hidx])
        w_birth = np.exp(w_birth - logsumexp(w_birth))

        # Extract cardinality distribution
        cdn_birth = np.zeros(n_birth.max() + 1)
        for card in range(cdn_birth.shape[0]):
            cdn_birth[card] = np.sum(w_birth[n_birth == card])

        # === Survive ===

        # Create surviving tracks
        N_surv = len(upd.track_table)
        tt_surv = [None for _ in range(N_surv)]
        for tabsidx in range(N_surv):
            mtemp_predict, Ptemp_predict = KalmanFilter.predict(self.model.motion_model.F,
                                                                self.model.motion_model.Q,
                                                                upd.track_table[tabsidx].gm.m,
                                                                upd.track_table[tabsidx].gm.P)

            w_surv = upd.track_table[tabsidx].gm.w.copy()
            m_surv = mtemp_predict.copy()
            P_surv = Ptemp_predict.copy()

            gm_surv = GaussianMixture(w_surv, m_surv, P_surv)
            l_surv = upd.track_table[tabsidx].l
            ah_surv = [x for x in upd.track_table[tabsidx].ah]

            tt_surv[tabsidx] = Track(gm_surv, l_surv, ah_surv)

        w_surv, I_surv, n_surv = [], [], []
        for pidx in range(len(upd.w)):
            if upd.n[pidx] == 0:  # TODO: check this case
                w_surv.append(np.log(upd.w[pidx]))
                I_surv.append([x for x in upd.I[pidx]])
                n_surv.append(upd.n[pidx].copy())
            else:
                # Calculate best surviving hypotheses
                pS = self.model.survival_model.get_probability()
                costv = pS / (1 - pS) * np.ones(upd.n[pidx])
                neglogcostv = -np.log(costv)
                N = int(
                    self.H_sur
                    * np.sqrt(upd.w[pidx]) / np.sum(np.sqrt(upd.w))
                    + 0.5
                )
                spaths, nlcost = kshortestwrap_pred(neglogcostv, N)

                # print(neglogcostv, N)
                # print(spaths, nlcost)
                # input()

                # Generate corresponding surviving hypotheses
                for hidx in range(len(nlcost)):
                    w_pd = upd.n[pidx] * np.log(1 - pS) \
                        + np.log(upd.w[pidx]) - nlcost[hidx]
                    I_pd = [upd.I[pidx][x] for x in spaths[hidx]]
                    n_pd = len(spaths[hidx])

                    w_surv.append(w_pd)
                    I_surv.append(I_pd)
                    n_surv.append(n_pd)
        w_surv = np.exp(w_surv - logsumexp(w_surv))
        n_surv = np.array(n_surv).astype(np.int32)
        # print(w_surv)
        # input()

        # Extract cardinality distribution
        cdn_surv = np.zeros(n_surv.max() + 1)
        for card in range(cdn_surv.shape[0]):
            cdn_surv[card] = np.sum(w_surv[n_surv == card])
        # print(cdn_surv)
        # input()

        tt_pred = tt_birth + tt_surv

        N = len(w_birth) * len(w_surv)
        w_pred = np.zeros(N)
        I_pred = [None for _ in range(N)]
        n_pred = np.zeros(N, dtype=np.int32)
        for bidx in range(len(w_birth)):
            for sidx in range(len(w_surv)):
                hidx = bidx * len(w_surv) + sidx
                w_pred[hidx] = w_birth[bidx] * w_surv[sidx]
                I_pred[hidx] = I_birth[bidx] + \
                    [x + len(tt_birth) for x in I_surv[sidx]]
                n_pred[hidx] = n_birth[bidx] + n_surv[sidx]
        w_pred = w_pred / w_pred.sum()

        # Extract cardinality distribution
        cdn_pred = np.zeros(n_pred.max() + 1)
        for card in range(cdn_pred.shape[0]):
            cdn_pred[card] = np.sum(w_pred[n_pred == card])

        pred = GLMB_GMS_Data(tt_pred, w_pred, I_pred, n_pred, cdn_pred)
        # print(pred.I)
        # input()
        pred = clean_predict(pred)
        # print(pred.w)
        # input()
        # input()

        return pred

    def gating(self,
               Z: np.ndarray,
               pred: GLMB_GMS_Data) -> np.ndarray:
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
               pred: GLMB_GMS_Data) -> GLMB_GMS_Data:
        # == Gating ==
        cand_Z = self.gating(Z, pred) \
            if self.use_gating \
            else Z
        # print(cand_Z)
        # input()

        # == Update ==

        # Create update tracks
        N1 = len(pred.track_table)
        N2 = cand_Z.shape[0]
        M = N1 * (N2 + 1)
        tt_upda = [None for _ in range(M)]

        # Missed detection tracks
        for tabidx in range(N1):
            gm = pred.track_table[tabidx].gm.copy()
            l = [x for x in pred.track_table[tabidx].l]
            ah = [x for x in pred.track_table[tabidx].ah] + [0]
            tt_upda[tabidx] = Track(gm, l, ah)
        # print(tt_upda)
        # input()

        # Updated tracks
        allcostm = np.zeros((N1, N2))
        for emm in range(N2):
            for tabidx in range(N1):
                qz_temp, m_temp, P_temp = KalmanFilter.update(cand_Z[[emm]],
                                                              self.model.measurement_model.H,
                                                              self.model.measurement_model.R,
                                                              pred.track_table[tabidx].gm.m,
                                                              pred.track_table[tabidx].gm.P)
                # print(qz_temp)
                # print(m_temp)
                # print(P_temp)
                # print('+++')
                w_temp = qz_temp[0] * pred.track_table[tabidx].gm.w + EPS
                # print(w_temp)
                # input('===')

                gm = GaussianMixture(w_temp / w_temp.sum(),
                                     m_temp[0], P_temp[[0]])
                l = [x for x in pred.track_table[tabidx].l]
                ah = pred.track_table[tabidx].ah + [emm]

                stoidx = N1 * (emm + 1) + (tabidx + 1) - 1
                tt_upda[stoidx] = Track(gm, l, ah)

                allcostm[tabidx, emm] = w_temp.sum()
        # print(allcostm)
        # input()

        # Component update
        lambda_c = self.model.clutter_model.lambda_c
        pdf_c = self.model.clutter_model.pdf_c
        pD = self.model.detection_model.get_probability()

        w_upda, I_upda, n_upda = [], [], []
        if N2 == 0:  # TODO: check this case
            w_upda = - lambda_c + pred.n * np.log(1 - pD) + np.log(pred.w)
            I_upda = [[xx for xx in x] for x in pred.I]
            n_upda = pred.n.copy()
        else:
            for pidx in range(len(pred.w)):
                if pred.n[pidx] == 0:
                    w_p = -lambda_c \
                        + N2 * np.log(lambda_c * pdf_c) \
                        + np.log(pred.w[pidx])
                    I_p = [x for x in pred.I[pidx]]
                    n_p = pred.n[pidx]

                    # print(w_p)
                    # print(I_p)
                    # print(n_p)
                    # input('===')

                    w_upda.append(w_p)
                    I_upda.append(I_p)
                    n_upda.append(n_p)
                else:
                    costm = pD / (1 - pD) * \
                        allcostm[pred.I[pidx]] / (lambda_c * pdf_c)
                    neglogcostm = - np.log(costm)

                    N = int(self.H_upd *
                            np.sqrt(pred.w[pidx]) / np.sum(np.sqrt(pred.w))
                            + 0.5)
                    uasses, nlcost = mbest_wrap(neglogcostm, N)

                    # if self.k == 2:
                    #     print(neglogcostm, N)
                    #     print(uasses, nlcost)
                    #     input()

                    for hidx in range(len(nlcost)):
                        w_ph = -lambda_c \
                            + N2 * np.log(lambda_c * pdf_c) \
                            + pred.n[pidx] * np.log(1 - pD) \
                            + np.log(pred.w[pidx]) \
                            - nlcost[hidx]
                        I_ph = [N1 * (x + 1) + (y + 1) - 1
                                for x, y in zip(uasses[hidx], pred.I[pidx])]
                        n_ph = pred.n[pidx]
                        # print(w_ph)
                        # print(I_ph)
                        # print(n_ph)
                        # input('===')

                        w_upda.append(w_ph)
                        I_upda.append(I_ph)
                        n_upda.append(n_ph)

        w_upda = np.exp(w_upda - logsumexp(w_upda))
        n_upda = np.array(n_upda)

        cdn_upda = np.zeros(n_upda.max() + 1)
        for card in range(cdn_upda.shape[0]):
            cdn_upda[card] = np.sum(w_upda[n_upda == card])
        # print(tt_upda)
        # print(w_upda)
        # print(I_upda)
        # print(n_upda)
        # print(cdn_upda)
        # input('====')

        upd = GLMB_GMS_Data(tt_upda, w_upda, I_upda, n_upda, cdn_upda)
        upd = clean_update(upd)
        # print(upd.track_table)
        # print(upd.w)
        # print(upd.I)
        # print(upd.n)
        # print(upd.cdn)
        # input('====')

        upd = prune(upd, self.hyp_thres)
        # print(upd.track_table)
        # print(upd.w)
        # print(upd.I)
        # print(upd.n)
        # print(upd.cdn)
        # input('====')
        upd = cap(upd, self.H_max)
        # print(upd.track_table)
        # print(upd.w)
        # print(upd.I)
        # print(upd.n)
        # print(upd.cdn)
        # input('====')

        return upd

    # def visualizable_estimate(self, upd: GLMB_GMS_Data) -> GLMB_GMS_Data:
    #     M = np.argmax(upd.cdn)
    #     T = [None for _ in range(M)]
    #     J = np.zeros((M, 2), dtype=np.int32)

    #     idxcmp = np.argmax(upd.w * (upd.n == M))
    #     for m in range(M):
    #         idxptr = upd.I[idxcmp][m]
    #         T[m] = [x for x in upd.track_table[idxptr].ah]
    #         J[m] = upd.track_table[idxptr].l

    #     H = [None for _ in range(M)]
    #     for m in range(M):
    #         H[m] = str(J[m, 0]) + '.' + str(J[m, 1])

    def visualizable_estimate(self, upd: GLMB_GMS_Data) -> GLMB_GMS_Data:
        M = np.argmax(upd.cdn)

        ah = [None for _ in range(M)]
        w = np.empty(M)
        X = np.empty((M, self.model.motion_model.x_dim))
        P = np.empty((M, self.model.motion_model.x_dim,
                     self.model.motion_model.x_dim))
        L = np.empty((M, 2), dtype=np.int32)

        idxcmp = np.argmax(upd.w * (upd.n == M))
        for m in range(M):
            idxtrk = np.argmax(upd.track_table[upd.I[idxcmp][m]].gm.w)
            w[m] = upd.track_table[upd.I[idxcmp][m]].gm.w[idxtrk].copy()
            X[m] = upd.track_table[upd.I[idxcmp][m]].gm.m[idxtrk].copy()
            P[m] = upd.track_table[upd.I[idxcmp][m]].gm.P[idxtrk].copy()
            L[m] = upd.track_table[upd.I[idxcmp][m]].l
            ah[m] = [x for x in upd.track_table[upd.I[idxcmp][m]].ah]

        return Track(GaussianMixture(w, X, P), L, ah)
