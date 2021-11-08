from dataclasses import dataclass
from typing import List, Tuple

from trackun.filters.base import GMSFilter
from trackun.common.gaussian_mixture import GaussianMixture
from trackun.common.kalman_filter import KalmanFilter
from trackun.common.gating import EllipsoidallGating
from trackun.common.murty import find_m_best_assignment

import numpy as np

__all__ = [
    'JointGLMB_GMS_Data',
    'JointGLMB_GMS_Filter',
]

EPS = np.finfo(np.float64).eps
INF = np.inf


def logsumexp(w):
    val = np.max(w)
    return np.log(np.sum(np.exp(w - val))) + val


def mbest_wrap(P0, m):
    assignments, costs = find_m_best_assignment(P0, m)
    costs = costs.reshape(-1)
    return assignments, costs


@dataclass
class Track:
    gm: GaussianMixture
    l: Tuple[int, int]
    ah: List[int]


@dataclass
class JointGLMB_GMS_Data:
    track_table: List[Track]
    w: np.ndarray
    I: List[np.ndarray]
    n: np.ndarray
    cdn: np.ndarray
    avps: np.ndarray


def hash_I(I):
    return '*'.join(map(lambda x: str(x+1), I)) + '*'


def clean_predict(w, I, n) -> JointGLMB_GMS_Data:
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
                 I: List[np.ndarray]) -> JointGLMB_GMS_Data:
    usedindicator = np.zeros(len(track_table), dtype=np.int64)
    for hidx in range(len(I)):
        usedindicator[I[hidx]] += 1
    trackcount = np.sum(usedindicator > 0)

    newindices = np.zeros(len(track_table), dtype=np.int64)
    newindices[usedindicator > 0] = np.arange(1, trackcount + 1)

    tt_temp = [x for x, y in zip(track_table, usedindicator) if y > 0]
    I_temp = [(newindices - 1)[I[hidx]] for hidx in range(len(I))]

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

    return JointGLMB_GMS_Data(tt_prune, w_prune, I_prune, n_prune, cdn_prune, None)


def prune(upd: JointGLMB_GMS_Data,
          hyp_threshold: float) -> JointGLMB_GMS_Data:
    idxkeep = np.where(upd.w > hyp_threshold)[0]
    return select(upd, idxkeep)


def cap(upd: JointGLMB_GMS_Data,
        H_max: int) -> JointGLMB_GMS_Data:
    if len(upd.w) > H_max:
        idxkeep = upd.w.argsort()[::-1][:H_max]
        return select(upd, idxkeep)
    else:
        return upd


class JointGLMB_GMS_Filter(GMSFilter):
    def __init__(self,
                 model,
                 L_max=100,
                 elim_thres=1e-5,
                 merge_threshold=4,
                 use_gating=True,
                 pG=0.9999999,
                 H_upd=3000,
                 H_max=3000,
                 hyp_thres=1e-15) -> None:
        super().__init__(model,
                         L_max, elim_thres, merge_threshold,
                         use_gating, pG)
        self.H_upd = H_upd
        self.H_max = H_max
        self.hyp_thres = hyp_thres

    def init(self):
        super().init()
        track_table = []
        w = np.ones(1)
        I = [np.empty(0, dtype=np.int64)]
        n = np.zeros(1).astype(np.int64)
        cdn = np.ones(1)
        return JointGLMB_GMS_Data(track_table, w, I, n, cdn, None)

    def predict(self, upd: JointGLMB_GMS_Data):
        # === Birth ===

        # Create birth tracks
        tt_birth = []
        for tabbidx in range(len(self.model.birth_model.rs)):
            gm = self.model.birth_model.gms[tabbidx]
            l = (self.k, tabbidx)
            ah = []
            tt_birth.append(Track(gm, l, ah))

        # === Survive ===

        # Create surviving tracks
        tt_surv = []
        for tabsidx in range(len(upd.track_table)):
            w_surv = upd.track_table[tabsidx].gm.w
            m_surv, P_surv = KalmanFilter.predict(self.model.motion_model.F,
                                                  self.model.motion_model.Q,
                                                  upd.track_table[tabsidx].gm.m,
                                                  upd.track_table[tabsidx].gm.P)
            gm_surv = GaussianMixture(w_surv, m_surv, P_surv)

            l_surv = upd.track_table[tabsidx].l
            ah_surv = upd.track_table[tabsidx].ah

            tt_surv.append(Track(gm_surv, l_surv, ah_surv))

        # Combine birth and surviving tracks
        tt_pred = tt_birth + tt_surv

        # Calculate average survival/death probabilites
        pb = self.model.birth_model.rs
        ps = self.model.survival_model.get_probability() * np.ones(len(upd.track_table))
        apvs = np.hstack([pb, ps])
        # print(apvs)
        # input()

        return JointGLMB_GMS_Data(tt_pred, upd.w, upd.I, upd.n, upd.cdn, apvs)

    def gating(self,
               Z: np.ndarray,
               pred: JointGLMB_GMS_Data) -> List[np.ndarray]:
        gate_indices = []
        if self.use_gating:
            for tabidx in range(len(pred.track_table)):
                gate_idx = EllipsoidallGating.filter_idx(Z, self.gamma,
                                                         self.model.measurement_model.H,
                                                         self.model.measurement_model.R,
                                                         pred.track_table[tabidx].gm.m,
                                                         pred.track_table[tabidx].gm.P)
                # print(gate_idx)
                # input()
                gate_indices.append(gate_idx)
        else:
            for tabidx in range(len(pred.track_table)):
                gate_indices.append(np.arange(len(Z)))
        return gate_indices

    def update(self, Z, pred: JointGLMB_GMS_Data):
        gate_indices = self.gating(Z, pred)

        # == Update ==
        avpd = self.model.detection_model.get_probability() * np.ones(len(pred.track_table))
        # print(avpd)
        # input('')

        # Create update tracks
        N1 = len(pred.track_table)
        N2 = Z.shape[0]
        M = N1 * (N2 + 1)
        tt_upda = [None for _ in range(M)]

        # Missed detection tracks
        for tabidx in range(N1):
            gm = pred.track_table[tabidx].gm
            l = pred.track_table[tabidx].l
            ah = pred.track_table[tabidx].ah + [0]
            tt_upda[tabidx] = Track(gm, l, ah)

        # Updated tracks
        allcostm = np.zeros((N1, N2))
        for tabidx in range(N1):
            qz_temp, m_temp, P_temp = KalmanFilter.update(Z[gate_indices[tabidx]],
                                                          self.model.measurement_model.H,
                                                          self.model.measurement_model.R,
                                                          pred.track_table[tabidx].gm.m,
                                                          pred.track_table[tabidx].gm.P)
            for i, emm in enumerate(gate_indices[tabidx]):
                w_temp = qz_temp[:, i] * pred.track_table[tabidx].gm.w + EPS

                gm = GaussianMixture(w_temp / w_temp.sum(),
                                     m_temp[:, i], P_temp)
                l = pred.track_table[tabidx].l
                ah = pred.track_table[tabidx].ah + [emm + 1]

                stoidx = N1 * (emm + 1) + (tabidx + 1) - 1
                tt_upda[stoidx] = Track(gm, l, ah)

                allcostm[tabidx, emm] = w_temp.sum()
                # print(tabidx, emm, allcostm[tabidx, emm])
                # input()
        # print(allcostm)
        # input()

        lambda_c = self.model.clutter_model.lambda_c
        pdf_c = self.model.clutter_model.pdf_c
        jointcostm = np.hstack([
            np.diag(1 - pred.avps),
            np.diag(pred.avps * (1 - avpd)),
            (pred.avps * avpd)[:, np.newaxis] * allcostm / (lambda_c * pdf_c)
        ])
        # print(jointcostm)
        # input()

        gatemeasidxs = np.full((N1, N2), -1)
        for tabidx in range(N1):
            gatemeasidxs[tabidx, :len(
                gate_indices[tabidx])] = gate_indices[tabidx]
        gatemeasindc = gatemeasidxs >= 0
        # print(gatemeasidxs, gatemeasindc)
        # input()

        # print(len(pred.w))

        w_upda, I_upda, n_upda = [], [], []
        w = -lambda_c + N2 * np.log(lambda_c * pdf_c)
        for pidx in range(len(pred.w)):
            cpreds = len(pred.track_table)
            nbirths = self.model.birth_model.N
            nexists = len(pred.I[pidx])
            ntracks = nbirths + nexists

            tindices = np.hstack([np.arange(nbirths), nbirths + pred.I[pidx]])

            lselmask = np.zeros((N1, N2), dtype=np.bool8)
            lselmask[tindices] = gatemeasindc[tindices]
            mindices = np.unique(gatemeasidxs[lselmask])

            costm = jointcostm[tindices][:, np.hstack([tindices,
                                                       cpreds + tindices,
                                                       2*cpreds + mindices])]
            neglogcostm = -np.log(costm)
            N = int(
                self.H_upd *
                np.sqrt(pred.w[pidx]) / np.sqrt(pred.w).sum()
                + 0.5
            )
            # print(neglogcostm)
            # print(N)
            # input()
            # uasses, nlcost = gibbs_wrap(neglogcostm, N)

            uasses, nlcost = mbest_wrap(neglogcostm, N)
            # print(uasses[-10:], nlcost[-10:])
            # input()
            # print(nlcost)
            # input()
            # print(uasses[:5])

            uasses[uasses < ntracks] = -1000
            uasses[np.logical_and(
                uasses >= ntracks, uasses < 2 * ntracks)] = -1
            uasses[uasses >= 2 * ntracks] = uasses[uasses >=
                                                   2 * ntracks] - 2 * ntracks
            uasses[uasses >= 0] = mindices[uasses[uasses >= 0]]

            # print(uasses[:5])
            # input()

            w_p = w + np.log(pred.w[pidx])
            for hidx in range(len(nlcost)):
                update_hypcmp_tmp = uasses[hidx]
                update_hypcmp_idx = cpreds * (update_hypcmp_tmp + 1) + tindices
                # print(cpreds)
                # print(tindices)
                # print(update_hypcmp_tmp)
                # print(update_hypcmp_idx)
                # input()

                w_ph = w_p - nlcost[hidx]
                I_ph = update_hypcmp_idx[update_hypcmp_idx >= 0]
                n_ph = (update_hypcmp_idx >= 0).sum()

                # if len(w_upda) > 235:
                #     print(uasses[hidx])
                #     print(update_hypcmp_tmp)
                #     print(update_hypcmp_idx)
                #     print(w_ph)
                #     print(I_ph)
                #     input()

                w_upda.append(w_ph)
                I_upda.append(I_ph)
                n_upda.append(n_ph)
        n_upda = np.array(n_upda).astype(np.int64)
        w_upda = np.exp(w_upda - logsumexp(w_upda))

        # print(I_upda)
        # input()

        cdn_upda = np.zeros(n_upda.max() + 1)
        for card in range(cdn_upda.shape[0]):
            cdn_upda[card] = np.sum(w_upda[n_upda == card])

        # print(cdn_upda)
        # input()

        w_upda, I_upda, n_upda = clean_predict(w_upda, I_upda, n_upda)
        tt_upda, I_upda = clean_update(tt_upda, I_upda)
        upd = JointGLMB_GMS_Data(
            tt_upda, w_upda, I_upda, n_upda, cdn_upda, None)

        # print(upd)
        # input()

        upd = prune(upd, self.hyp_thres)
        upd = cap(upd, self.H_max)

        # print(upd.cdn)
        # input()

        return upd

    def visualizable_estimate(self, upd: JointGLMB_GMS_Data) -> Track:
        M = np.argmax(upd.cdn)

        ah = [None for _ in range(M)]
        w = np.empty(M)
        X = np.empty((M, self.model.motion_model.x_dim))
        P = np.empty((M, self.model.motion_model.x_dim,
                     self.model.motion_model.x_dim))
        L = np.empty((M, 2), dtype=np.int64)

        idxcmp = np.argmax(upd.w * (upd.n == M))
        for m in range(M):
            w[m], X[m], P[m] = \
                upd.track_table[upd.I[idxcmp][m]].gm.cap(1).unpack()
            L[m] = upd.track_table[upd.I[idxcmp][m]].l
            ah[m] = upd.track_table[upd.I[idxcmp][m]].ah

        return Track(GaussianMixture(w, X, P), L, ah)

    def estimate(self, upd: JointGLMB_GMS_Data):
        M = np.argmax(upd.cdn)
        X = np.empty((M, self.model.motion_model.x_dim))
        L = np.empty((M, 2), dtype=np.int64)

        idxcmp = np.argmax(upd.w * (upd.n == M))
        for m in range(M):
            _, X[m], _ = \
                upd.track_table[upd.I[idxcmp][m]].gm.cap(1).unpack()
            L[m] = upd.track_table[upd.I[idxcmp][m]].l

        return X, L
