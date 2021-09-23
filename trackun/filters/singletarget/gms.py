from dataclasses import dataclass

from trackun.filters.base import GMSFilter
from trackun.common.gaussian_mixture import GaussianMixture
from trackun.common.kalman_filter import KalmanFilter
from trackun.common.gating import EllipsoidallGating

import numpy as np
from scipy.stats.distributions import chi2

__all__ = [
    'KF_SingleTarget_Data',
    'SingleTarget_GMS_Filter',
]


@dataclass
class KF_SingleTarget_Data:
    r: float
    gm: GaussianMixture


class SingleTarget_GMS_Filter(GMSFilter):
    def __init__(self, model,
                 L_max=100,
                 elim_thres=1e-5,
                 merge_threshold=4,
                 use_gating=True,
                 pG=0.999) -> None:
        self.model = model

        self.L_max = L_max
        self.elim_threshold = elim_thres
        self.merge_threshold = merge_threshold

        self.use_gating = use_gating
        self.gamma = chi2.ppf(pG, self.model.z_dim)

    def init(self):
        w_upds_k = np.array([1.])
        m_upds_k = np.zeros((1, self.model.x_dim))
        P_upds_k = np.eye(self.model.x_dim)[np.newaxis, :]

        gm_upds_k = GaussianMixture(w_upds_k, m_upds_k, P_upds_k)
        return KF_SingleTarget_Data(gm_upds_k)

    def predict(self, upds_k):
        w_preds_k = upds_k.gm.w.copy()
        m_preds_k, P_preds_k = \
            KalmanFilter.predict(self.model.motion_model.F,
                                 self.model.motion_model.Q,
                                 upds_k.gm.m, upds_k.gm.P)

        gm_preds_k = GaussianMixture(w_preds_k, m_preds_k, P_preds_k)
        return KF_SingleTarget_Data(gm_preds_k)

    def gating(self, Z, preds_k):
        return EllipsoidallGating.filter(Z,
                                         self.gamma,
                                         self.model.measurement_model.H,
                                         self.model.measurement_model.R,
                                         preds_k.gm.m, preds_k.gm.P)

    def postprocess(self, gm_ups_k):
        gm_ups_k = gm_ups_k.prune(self.elim_threshold)
        gm_ups_k = gm_ups_k.merge_and_cap(self.merge_threshold, self.L_max)
        return gm_ups_k

    def update(self, Z, preds_k):
        # == Gating ==
        cand_Z = self.gating(Z, preds_k) \
            if self.use_gating \
            else Z

        # == Update ==
        N1 = preds_k.gm.w.shape[0]
        N2 = cand_Z.shape[0]
        M = N1 * (N2 + 1)

        gm_upds_k = GaussianMixture.get_empty(M, self.model.x_dim)

        # Miss detection
        gm_upds_k.w[:N1] = preds_k.gm.w \
            * (1 - self.model.detection_model.get_probability()) \
            * self.model.clutter_model.rate_c
        gm_upds_k.m[:N1] = preds_k.gm.m.copy()
        gm_upds_k.P[:N1] = preds_k.gm.P.copy()

        # Detection
        if N2 > 0:
            qs, ms, Ps = KalmanFilter.update(cand_Z,
                                             self.model.measurement_model.H,
                                             self.model.measurement_model.R,
                                             preds_k.gm.m, preds_k.gm.P)

            gm_upds_k.w[N1:] = (self.model.detection_model.get_probability()
                                * preds_k.gm.w * qs.T).reshape(-1)
            gm_upds_k.m[N1:] = ms.transpose(1, 0, 2).reshape(N1 * N2, -1)
            gm_upds_k.P[N1:] = np.tile(Ps, (N2, 1, 1))

        gm_upds_k.w = gm_upds_k.w / gm_upds_k.w.sum()

        # == Post-processing ==
        gm_upds_k = self.postprocess(gm_upds_k)

        return KF_SingleTarget_Data(gm_upds_k)

    def visualizable_estimate(self, upds_k):
        idx = np.argmax(upds_k.gm.w)
        gm_ests_k = upds_k.gm.select(idx)
        return KF_SingleTarget_Data(upds_k.r, gm_ests_k)

    def estimate(self, upds_k):
        idx = np.argmax(upds_k.gm.w)
        return upds_k.gm.select(idx).m
