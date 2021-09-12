from dataclasses import dataclass

from trackun.common.gaussian_mixture import GaussianMixture
from trackun.common.kalman_filter import KalmanFilter
from trackun.common.gating import EllipsoidallGating

import numpy as np
from scipy.stats.distributions import chi2

__all__ = [
    'KF_PHD_Data',
    'PHD_GMS_Filter',
]


@dataclass
class KF_PHD_Data:
    gm: GaussianMixture


class PHD_GMS_Filter:
    def __init__(self,
                 model,
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
        w = np.array([1.])
        m = np.zeros((1, self.model.x_dim))
        P = np.eye(self.model.x_dim)[np.newaxis, :]

        gm = GaussianMixture(w, m, P)
        return KF_PHD_Data(gm)

    def predict(self, upds_k):
        N = upds_k.gm.w.shape[0]
        L = self.model.birth_model.N

        w_preds_k, m_preds_k, P_preds_k = \
            GaussianMixture.get_empty(N+L, self.model.x_dim).unpack()

        # Predict surviving states
        w_preds_k[L:] = \
            self.model.survival_model.get_probability() * upds_k.gm.w
        m_preds_k[L:], P_preds_k[L:] = \
            KalmanFilter.predict(self.model.motion_model.F,
                                 self.model.motion_model.Q,
                                 upds_k.gm.m, upds_k.gm.P)

        # Predict born states
        w_preds_k[:L] = self.model.birth_model.ws
        m_preds_k[:L] = self.model.birth_model.ms
        P_preds_k[:L] = self.model.birth_model.Ps

        gm_preds_k = GaussianMixture(w_preds_k, m_preds_k, P_preds_k)
        return KF_PHD_Data(gm_preds_k)

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

        w_upds_k, m_upds_k, P_upds_k = \
            GaussianMixture.get_empty(M, self.model.x_dim).unpack()

        # Miss detection
        w_upds_k[:N1] = preds_k.gm.w \
            * (1 - self.model.detection_model.get_probability())
        m_upds_k[:N1] = preds_k.gm.m.copy()
        P_upds_k[:N1] = preds_k.gm.P.copy()

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
            w_upds_k[N1:] = w.reshape(-1)

            m_upds_k[N1:] = \
                ms.transpose(1, 0, 2).reshape(-1, self.model.x_dim)
            P_upds_k[N1:] = np.tile(Ps, (N2, 1, 1))

        # == Post-processing ==
        gm_upds_k = GaussianMixture(w_upds_k, m_upds_k, P_upds_k)
        gm_upds_k = self.postprocess(gm_upds_k)

        return KF_PHD_Data(gm_upds_k)

    def estimate(self, upds_k):
        cnt = upds_k.gm.w.round().astype(np.int32)
        w_ests_k = upds_k.gm.w.repeat(cnt, axis=0)
        m_ests_k = upds_k.gm.m.repeat(cnt, axis=0)
        P_ests_k = upds_k.gm.P.repeat(cnt, axis=0)

        gm_ests_k = GaussianMixture(w_ests_k, m_ests_k, P_ests_k)
        return KF_PHD_Data(gm_ests_k)

    def step(self, Z, upds_k):
        # == Predict ==
        preds_k = self.predict(upds_k)

        # == Update ==
        upds_k = self.update(Z, preds_k)

        return upds_k

    def run(self, Zs):
        # Initialize
        upds_k = self.init()

        # Recursive loop
        ests = []
        for Z in Zs:
            upds_k = self.step(Z, upds_k)
            ests_k = self.estimate(upds_k)
            ests.append(ests_k)

        return [est.gm.w for est in ests],\
            [est.gm.m for est in ests],\
            [est.gm.P for est in ests]
