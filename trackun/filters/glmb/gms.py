from dataclasses import dataclass
from typing import List

from trackun.filters.base import GMSFilter
from trackun.common.gaussian_mixture import GaussianMixture
from trackun.common.kalman_filter import KalmanFilter
from trackun.common.gating import EllipsoidallGating
from trackun.common.kshortest import kshortestwrap_pred

import numpy as np

__all__ = [
    'GLMB_GMS_Data',
    'GLMB_GMS_Filter',
]


@dataclass
class Track:
    gm: GaussianMixture
    l: int
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
        _hash[hidx] = '*'.join(map(str, sorted(pred.I[hidx]))) + '*'

    cu, _, ic = np.unique(_hash,
                          return_index=True,
                          return_inverse=True)

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

    def init(self):
        track_table = []
        w = np.ones(1)
        I = [[]]
        n = np.zeros(1).astype(np.int32)
        cdn = np.ones(1)
        return GLMB_GMS_Data(track_table, w, I, n, cdn)

    def predict(self,
                upd: GLMB_GMS_Data):
        # === Birth ===

        # Create birth tracks
        N_birth = len(self.model.birth_model.rs)
        tt_birth = [None for _ in range(N_birth)]
        for tabbidx in range(N_birth):
            gm = self.model.birth_model.gms[tabbidx].copy()
            l = tabbidx
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
            m_surv = mtemp_predict.copy()
            P_surv = Ptemp_predict.copy()
            w_surv = upd.track_table[tabsidx].gm.w.copy()

            gm_surv = GaussianMixture(w_surv, m_surv, P_surv)
            l_surv = upd.track_table[tabsidx].l
            ah_surv = [x for x in upd.track_table[tabsidx].ah]

            tt_surv[tabsidx] = Track(gm_surv, l_surv, ah_surv)

        w_surv, I_surv, n_surv = [], [], []
        runidx = 1
        for pidx in range(len(upd.w)):
            if upd.n[pidx] == 0:
                w_surv.append(np.log(upd.w[pidx]))
                I_surv.append([[xx for xx in x] for x in upd.I[pidx]])
                n_surv.append(upd.n[pidx].copy())
                runidx += 1
            else:
                # Calculate best surviving hypotheses
                pS = self.model.survival_model.get_probability()
                costv = pS / (1 - pS) * np.ones(upd.n[pidx])
                neglogcostv = -np.log(costv)

                N = int(
                    self.H_sur
                    * np.sqrt(upd.w[pidx]) / np.sum(np.sqrt(upd.w[pidx]))
                )
                spaths, nlcost = kshortestwrap_pred(neglogcostv, N)

                # Generate corresponding surviving hypotheses
                for hidx in range(len(nlcost)):
                    w_surv.append(
                        upd.n[pidx] * np.log(1 - pS)
                        + np.log(upd.w[pidx]) - nlcost[hidx])
                    I_surv.append(upd.I[pidx][spaths[hidx]])
                    n_surv.append(len(spaths[hidx]))
                    runidx += 1
        w_surv = np.exp(w_surv - logsumexp(w_surv))
        n_surv = np.array(n_surv).astype(np.int32)

        # Extract cardinality distribution
        cdn_surv = np.zeros(n_surv.max() + 1)
        for card in range(cdn_surv.shape[0]):
            cdn_surv[card] = np.sum(w_surv[n_surv == card])

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
        pred = clean_predict(pred)
        print(pred)

        return pred
