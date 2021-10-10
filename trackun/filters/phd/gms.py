from dataclasses import dataclass

from trackun.filters.base import GMSFilter
from trackun.common.gaussian_mixture import GaussianMixture
from trackun.common.kalman_filter import KalmanFilter
from trackun.common.gating import EllipsoidallGating

import numpy as np

__all__ = [
    'PHD_GMS_Data',
    'PHD_GMS_Filter',
]


@dataclass
class PHD_GMS_Data:
    gm: GaussianMixture


class PHD_GMS_Filter(GMSFilter):
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
        super().init()
        w = np.array([1.])
        m = np.zeros((1, self.model.x_dim))
        P = np.eye(self.model.x_dim)[np.newaxis, :]

        gm = GaussianMixture(w, m, P)
        return PHD_GMS_Data(gm)

    def predict(self, upds_k):
        # Predict born states
        w_bir, m_bir, P_bir = \
            self.model.birth_model.get_birth_sites()

        # Predict surviving state
        w_sur = self.model.survival_model.get_probability() * upds_k.gm.w
        m_sur, P_sur = KalmanFilter.predict(self.model.motion_model.F,
                                            self.model.motion_model.Q,
                                            upds_k.gm.m, upds_k.gm.P)

        w_preds_k = np.hstack([w_bir, w_sur])
        m_preds_k = np.vstack([m_bir, m_sur])
        P_preds_k = np.vstack([P_bir, P_sur])
        gm_preds_k = GaussianMixture(w_preds_k, m_preds_k, P_preds_k)

        return PHD_GMS_Data(gm_preds_k)

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
            * (1 - self.model.detection_model.get_probability())
        gm_upds_k.m[:N1] = preds_k.gm.m.copy()
        gm_upds_k.P[:N1] = preds_k.gm.P.copy()

        # Detection
        if N2 > 0:
            qs, ms, Ps = KalmanFilter.update(cand_Z,
                                             self.model.measurement_model.H,
                                             self.model.measurement_model.R,
                                             preds_k.gm.m, preds_k.gm.P)

            w = (preds_k.gm.w * qs.T) \
                * self.model.detection_model.get_probability()
            w = w / (self.model.clutter_model.lambda_c
                     * self.model.clutter_model.pdf_c
                     + w.sum(1)[:, np.newaxis])
            gm_upds_k.w[N1:] = w.reshape(-1)

            gm_upds_k.m[N1:] = \
                ms.transpose(1, 0, 2).reshape(-1, self.model.x_dim)
            gm_upds_k.P[N1:] = np.tile(Ps, (N2, 1, 1))

        # == Post-processing ==
        gm_upds_k = self.postprocess(gm_upds_k)

        return PHD_GMS_Data(gm_upds_k)

    def visualizable_estimate(self, upds_k):
        cnt = upds_k.gm.w.round().astype(np.int32)
        w_ests_k = upds_k.gm.w.repeat(cnt, axis=0)
        m_ests_k = upds_k.gm.m.repeat(cnt, axis=0)
        P_ests_k = upds_k.gm.P.repeat(cnt, axis=0)
        gm_ests_k = GaussianMixture(w_ests_k, m_ests_k, P_ests_k)
        return PHD_GMS_Data(gm_ests_k)

    def estimate(self, upds_k):
        cnt = upds_k.gm.w.round().astype(np.int32)
        m_ests_k = upds_k.gm.m.repeat(cnt, axis=0)
        return m_ests_k
