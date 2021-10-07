from dataclasses import dataclass

from trackun.filters.base import GMSFilter
from trackun.common.gaussian_mixture import GaussianMixture
from trackun.common.kalman_filter import KalmanFilter
from trackun.common.gating import EllipsoidallGating

import numpy as np

__all__ = [
    'Bernoulli_GMS_Data',
    'Bernoulli_GMS_Filter',
]


@dataclass
class Bernoulli_GMS_Data:
    r: float
    gm: GaussianMixture


class Bernoulli_GMS_Filter(GMSFilter):
    def __init__(self,
                 model,
                 L_max=100,
                 elim_thres=1e-5,
                 merge_threshold=4,
                 use_gating=True,
                 pG=0.999) -> None:
        super().__init__(model,
                         L_max, elim_thres, merge_threshold,
                         use_gating, pG)

    def init(self):
        r_upds_k = 0.
        w_upds_k = np.array([1.])
        m_upds_k = np.zeros((1, self.model.x_dim))
        P_upds_k = np.eye(self.model.x_dim)[np.newaxis, :]

        gm_upds_k = GaussianMixture(w_upds_k, m_upds_k, P_upds_k)
        return Bernoulli_GMS_Data(r_upds_k, gm_upds_k)

    def predict(self, upds_k):
        r_preds_k = (1 - upds_k.r) * self.model.birth_model.rs \
            + upds_k.r * self.model.survival_model.get_probability()

        N = upds_k.gm.w.shape[0]
        L = self.model.birth_model.Ls[0]

        w_preds_k, m_preds_k, P_preds_k = \
            GaussianMixture.get_empty(N+L, self.model.x_dim).unpack()

        # Predict surviving states
        w_preds_k[L:] = upds_k.gm.w * upds_k.r \
            * self.model.survival_model.get_probability()
        m_preds_k[L:], P_preds_k[L:] = \
            KalmanFilter.predict(self.model.motion_model.F,
                                 self.model.motion_model.Q,
                                 upds_k.gm.m, upds_k.gm.P)

        # Predict born states
        w_preds_k[:L] = self.model.birth_model.gms[0].w \
            * self.model.birth_model.rs * (1 - upds_k.r)
        m_preds_k[:L] = self.model.birth_model.gms[0].m
        P_preds_k[:L] = self.model.birth_model.gms[0].P

        # Normalize prediction
        w_preds_k = w_preds_k / w_preds_k.sum()

        gm_preds_k = GaussianMixture(w_preds_k, m_preds_k, P_preds_k)
        return Bernoulli_GMS_Data(r_preds_k, gm_preds_k)

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

        w_ups_k_sum = gm_upds_k.w.sum()
        r_upds_k = preds_k.r * w_ups_k_sum / (
            (1 - preds_k.r)
            * self.model.clutter_model.rate_c
            + preds_k.r * w_ups_k_sum
        )
        gm_upds_k.w = gm_upds_k.w / w_ups_k_sum

        # == Post-processing ==
        gm_upds_k = self.postprocess(gm_upds_k)

        return Bernoulli_GMS_Data(r_upds_k, gm_upds_k)

    def visualizable_estimate(self, upds_k):
        idx = [] \
            if upds_k.r <= 0.5 or len(upds_k.gm.w) == 0 \
            else [np.argmax(upds_k.gm.w)]
        gm_ests_k = upds_k.gm.select(idx)
        return Bernoulli_GMS_Data(upds_k.r, gm_ests_k)

    def estimate(self, upds_k):
        idx = [] \
            if upds_k.r <= 0.5 and len(upds_k.gm.w) == 0 \
            else [np.argmax(upds_k.gm.w)]
        return upds_k.gm.select(idx).m
