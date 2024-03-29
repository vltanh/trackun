from dataclasses import dataclass
from typing import List, Tuple

from trackun.filters.base import GMSFilter
from trackun.common.gaussian_mixture import GaussianMixture
from trackun.common.kalman_filter import KalmanFilter
from trackun.common.gating import EllipsoidallGating
from trackun.common.murty import find_m_best_assignment

import numpy as np

__all__ = [
    'JointLMB_GMS_Data',
    'JointLMB_GMS_Filter',
]

EPS = np.finfo(np.float64).eps
INF = np.inf


@dataclass
class Track:
    gm: GaussianMixture
    l: Tuple[int, int]
    ah: List[int]


@dataclass
class JointLMB_GMS_Data:
    track_table: List[Track]
    r: np.ndarray
    w: np.ndarray
    I: List[np.ndarray]
    n: np.ndarray
    cdn: np.ndarray
    avps: np.ndarray
    N: int


@dataclass
class LMB:
    track_table: List[Track]
    r: np.ndarray


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


def mbest_wrap(P0, m):
    assignments, costs = find_m_best_assignment(P0, m)
    costs = costs.reshape(-1)
    return assignments, costs


def hash_I(I):
    return '*'.join(map(lambda x: str(x+1), I)) + '*'


def clean_predict(w, I, n):
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
                 I: List[np.ndarray]):
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


class JointLMB_GMS_Filter(GMSFilter):
    def __init__(self,
                 model,
                 L_max=100,
                 elim_thres=1e-5,
                 merge_threshold=4,
                 use_gating=True,
                 pG=0.9999999,
                 T_max=100,
                 track_threshold=1e-3,
                 H_upd=50) -> None:
        super().__init__(model,
                         L_max, elim_thres, merge_threshold,
                         use_gating, pG)
        self.T_max = T_max
        self.track_threshold = track_threshold

        self.H_upd = H_upd

    def init(self):
        super().init()
        track_table = []
        r = np.empty(0)
        w = np.ones(1)
        I = [np.empty(0, dtype=np.int64)]
        n = np.zeros(1).astype(np.int64)
        cdn = np.ones(1)
        return JointLMB_GMS_Data(track_table, r, w, I, n, cdn, None, None)

    def predict(self, upd: JointLMB_GMS_Data):
        # === Birth ===

        # Create birth tracks

        tt_birth = []
        for tabbidx in range(len(self.model.birth_model.rs)):
            gm = self.model.birth_model.gms[tabbidx]
            l = (self.k, tabbidx)
            ah = []
            tt_birth.append(Track(gm, l, ah))

        # === Survive ===

        pS = self.model.survival_model.get_probability()

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
        r_birth = self.model.birth_model.rs
        r_surv = pS * upd.r
        apvs = np.hstack([r_birth, r_surv])
        # print(apvs)
        # input()

        # print(tt_pred)
        # input()

        return JointLMB_GMS_Data(tt_pred, upd.r, upd.w, upd.I, upd.n, upd.cdn, apvs, len(upd.track_table))

    def gating(self,
               Z: np.ndarray,
               pred: JointLMB_GMS_Data) -> List[np.ndarray]:
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

    def update(self, Z, pred: JointLMB_GMS_Data):
        gate_indices = self.gating(Z, pred)
        # print(Z)
        # print(gate_indices)
        # input()

        # == Update ==
        avpd = self.model.detection_model.get_probability() * np.ones(len(pred.track_table))
        # print(avpd)
        # input()

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
            # print(qz_temp)
            for i, emm in enumerate(gate_indices[tabidx]):
                w_temp = qz_temp[:, i] * pred.track_table[tabidx].gm.w + EPS

                gm = GaussianMixture(w_temp / w_temp.sum(),
                                     m_temp[:, i], P_temp)
                l = pred.track_table[tabidx].l
                ah = pred.track_table[tabidx].ah + [emm + 1]

                stoidx = N1 * (emm + 1) + (tabidx + 1) - 1
                tt_upda[stoidx] = Track(gm, l, ah)

                allcostm[tabidx, emm] = w_temp.sum()
            # print(allcostm[tabidx])
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

        cpreds = len(pred.track_table)
        nbirths = self.model.birth_model.N
        nexists = pred.N
        ntracks = nbirths + nexists

        tindices = np.hstack(
            [np.arange(nbirths), nbirths + np.arange(nexists)])

        lselmask = np.zeros((N1, N2), dtype=np.bool8)
        lselmask[tindices] = gatemeasindc[tindices]
        mindices = np.unique(gatemeasidxs[lselmask])

        costm = jointcostm[tindices][:, np.hstack([tindices,
                                                   cpreds + tindices,
                                                   2*cpreds + mindices])]
        neglogcostm = -np.log(costm)
        # print(neglogcostm)
        # input()

        N = int(self.H_upd)
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

        # print(uasses)
        # print(nlcost)
        # input()

        for hidx in range(len(nlcost)):
            update_hypcmp_tmp = uasses[hidx]
            update_hypcmp_idx = cpreds * (update_hypcmp_tmp + 1) + tindices
            # print(cpreds)
            # print(tindices)
            # print(update_hypcmp_tmp)
            # print(update_hypcmp_idx)
            # input()

            w_ph = w - nlcost[hidx]
            I_ph = update_hypcmp_idx[update_hypcmp_idx >= 0]
            n_ph = (update_hypcmp_idx >= 0).sum()

            # if len(w_upda) > 235:
            # print(uasses[hidx])
            # print(update_hypcmp_tmp)
            # print(update_hypcmp_idx)
            # print(w_ph)
            # print(I_ph)
            # input()

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

        # print(w_upda)
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

        upd = JointLMB_GMS_Data(upd.track_table, upd.r,
                                None, None, None, None, None, None)

        return upd

    def visualizable_estimate(self, upd: JointLMB_GMS_Data) -> Track:
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

    def estimate(self, upd: JointLMB_GMS_Data):
        cdn = np.prod(1 - upd.r) * esf(upd.r / (1 - upd.r))
        M = np.argmax(cdn)
        M = min(len(upd.r), M)

        X = np.empty((M, self.model.motion_model.x_dim))
        L = np.empty((M, 2), dtype=np.int64)

        idxcmp = upd.r.argsort()[::-1]
        for m in range(M):
            track = upd.track_table[idxcmp[m]]
            _, X[m], _ = track.gm.cap(1).unpack()
            L[m] = track.l

        return X, L
