from dataclasses import dataclass
from typing import List

from trackun.filters.base import GMSFilter
from trackun.common.gaussian_mixture import GaussianMixture
from trackun.common.kalman_filter import KalmanFilter
from trackun.common.gating import EllipsoidallGating
from trackun.common.kshortest import find_k_shortest_path
from trackun.common.murty import find_m_best_assignment

import numpy as np

__all__ = [
    'GLMB_GMS_Data',
    'GLMB_GMS_Filter',
]

EPS = np.finfo(np.float64).eps
INF = np.inf


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
            paths[p] = _is[paths[p]].tolist()
    return paths, costs


def logsumexp(w):
    if np.all(w == -np.inf):
        return -np.inf
    val = np.max(w)
    return np.log(np.sum(np.exp(w - val))) + val


def hash_I(I):
    return '*'.join(map(lambda x: str(x+1), I)) + '*'


def clean_predict(w, I, n) -> GLMB_GMS_Data:
    _hash = [hash_I(sorted(I[hidx]))
             for hidx in range(len(w))]

    _, ia, ic = np.unique(_hash,
                          return_index=True,
                          return_inverse=True)

    w_temp = np.zeros(len(ia))
    for hidx in range(len(ic)):
        w_temp[ic[hidx]] = w_temp[ic[hidx]] + w[hidx]

    I_temp = [I[hidx] for hidx in ia]
    n_temp = n[ia]

    return w_temp, I_temp, n_temp


def clean_update(track_table: List[Track],
                 I: List[List[int]]) -> GLMB_GMS_Data:
    usedindicator = np.zeros(len(track_table), dtype=np.int32)
    for hidx in range(len(I)):
        usedindicator[I[hidx]] += 1
    trackcount = np.sum(usedindicator > 0)

    newindices = np.zeros(len(track_table), dtype=np.ndarray)
    newindices[usedindicator > 0] = np.arange(1, trackcount + 1)

    tt_temp = [x for x, y in zip(track_table, usedindicator) if y > 0]
    I_temp = [(newindices - 1)[I[hidx]].copy().tolist()
              for hidx in range(len(I))]

    return tt_temp, I_temp


def select(upd, idxkeep):
    tt_prune = upd.track_table
    w_prune = upd.w[idxkeep]
    I_prune = [upd.I[x] for x in idxkeep]
    n_prune = upd.n[idxkeep]

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
        tt_birth = []
        for tabbidx in range(N_birth):
            gm = self.model.birth_model.gms[tabbidx]
            l = [self.k, tabbidx]
            ah = []
            tt_birth.append(Track(gm, l, ah))

        # Calculate best birth hypotheses
        r_birth = self.model.birth_model.rs
        costv = r_birth / (1 - r_birth)
        neglogcostv = -np.log(costv)
        bpaths, nlcost = kshortest_wrap(neglogcostv, self.H_bth)

        # Generate corresponding birth hypotheses
        w_birth = np.empty(len(nlcost))
        I_birth = []
        n_birth = np.empty(len(nlcost), dtype=np.int32)
        for hidx in range(len(nlcost)):
            w_birth[hidx] = np.log(1 - r_birth).sum() - nlcost[hidx]
            I_birth.append(bpaths[hidx])
            n_birth[hidx] = len(bpaths[hidx])
        w_birth = np.exp(w_birth - logsumexp(w_birth))

        # === Survive ===

        # Create surviving tracks
        N_surv = len(upd.track_table)
        tt_surv = [None for _ in range(N_surv)]
        for tabsidx in range(N_surv):
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

            tt_surv[tabsidx] = Track(gm_surv, l_surv, ah_surv)

        pS = self.model.survival_model.get_probability()

        w_surv, I_surv, n_surv = [], [], []
        for pidx in range(len(upd.w)):
            if upd.n[pidx] == 0:
                w_surv.append(np.log(upd.w[pidx]))
                I_surv.append(upd.I[pidx])
                n_surv.append(upd.n[pidx])
            else:
                # Calculate best surviving hypotheses
                N = int(
                    self.H_sur * np.sqrt(upd.w[pidx]) / np.sum(np.sqrt(upd.w))
                    + 0.5
                )

                if N > 0:
                    costv = pS / (1 - pS) * np.ones(upd.n[pidx])
                    neglogcostv = -np.log(costv)

                    spaths, nlcost = kshortest_wrap(neglogcostv, N)

                    # Generate corresponding surviving hypotheses
                    w_p = upd.n[pidx] * np.log(1 - pS) + np.log(upd.w[pidx])
                    for hidx in range(len(nlcost)):
                        w_pd = w_p - nlcost[hidx]
                        I_pd = [upd.I[pidx][x] for x in spaths[hidx]]
                        n_pd = len(spaths[hidx])

                        w_surv.append(w_pd)
                        I_surv.append(I_pd)
                        n_surv.append(n_pd)
        w_surv = np.exp(w_surv - logsumexp(w_surv))
        n_surv = np.array(n_surv).astype(np.int32)

        # Combine birth and surviving tracks
        tt_pred = tt_birth + tt_surv

        w_pred = (w_birth[:, np.newaxis] * w_surv[np.newaxis, :]).reshape(-1)
        w_pred = w_pred / w_pred.sum()

        I_pred = [
            I_birth[bidx] + [x + len(tt_birth) for x in I_surv[sidx]]
            for bidx in range(len(w_birth))
            for sidx in range(len(w_surv))
        ]

        n_pred = (n_birth[:, np.newaxis] + n_surv[np.newaxis, :]).reshape(-1)

        cdn_pred = np.zeros(n_pred.max() + 1)
        for card in range(cdn_pred.shape[0]):
            cdn_pred[card] = np.sum(w_pred[n_pred == card])

        w_pred, I_pred, n_pred = clean_predict(w_pred, I_pred, n_pred)
        pred = GLMB_GMS_Data(tt_pred, w_pred, I_pred, n_pred, cdn_pred)

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

        if N2 == 0:  # TODO: check this case
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
                            I_ph = [N1 * (x + 1) + (y + 1) - 1
                                    for x, y in zip(uasses[hidx], pred.I[pidx])]
                            n_ph = pred.n[pidx]

                            w_upda.append(w_ph)
                            I_upda.append(I_ph)
                            n_upda.append(n_ph)
            n_upda = np.array(n_upda).astype(np.int32)
        w_upda = np.exp(w_upda - logsumexp(w_upda))

        cdn_upda = np.zeros(n_upda.max() + 1)
        for card in range(cdn_upda.shape[0]):
            cdn_upda[card] = np.sum(w_upda[n_upda == card])

        tt_upda, I_upda = clean_update(tt_upda, I_upda)
        upd = GLMB_GMS_Data(tt_upda, w_upda, I_upda, n_upda, cdn_upda)

        upd = prune(upd, self.hyp_thres)
        upd = cap(upd, self.H_max)

        return upd

    def visualizable_estimate(self, upd: GLMB_GMS_Data) -> Track:
        M = np.argmax(upd.cdn)

        ah = [None for _ in range(M)]
        w = np.empty(M)
        X = np.empty((M, self.model.motion_model.x_dim))
        P = np.empty((M, self.model.motion_model.x_dim,
                     self.model.motion_model.x_dim))
        L = np.empty((M, 2), dtype=np.int32)

        idxcmp = np.argmax(upd.w * (upd.n == M))
        for m in range(M):
            w[m], X[m], P[m] = \
                upd.track_table[upd.I[idxcmp][m]].gm.cap(1).unpack()
            L[m] = upd.track_table[upd.I[idxcmp][m]].l
            ah[m] = upd.track_table[upd.I[idxcmp][m]].ah

        return Track(GaussianMixture(w, X, P), L, ah)

    def estimate(self, upd: GLMB_GMS_Data):
        M = np.argmax(upd.cdn)

        X = np.empty((M, self.model.motion_model.x_dim))
        L = np.empty((M, 2), dtype=np.int32)

        idxcmp = np.argmax(upd.w * (upd.n == M))
        for m in range(M):
            idxtrk = np.argmax(upd.track_table[upd.I[idxcmp][m]].gm.w)
            X[m] = upd.track_table[upd.I[idxcmp][m]].gm.m[idxtrk]
            L[m] = upd.track_table[upd.I[idxcmp][m]].l

        return X, L
