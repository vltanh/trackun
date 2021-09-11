from dataclasses import dataclass

from trackun.common.kalman import kalman_predict, kalman_update
from trackun.common.hypotheses_reduction import prune, merge_and_cap
from trackun.common.gating import gate

import numpy as np
from scipy.stats.distributions import chi2

__all__ = ['Bernoulli_GMS_Filter']


@dataclass
class KF_Bernoulli_Data:
    r: float
    w: np.ndarray
    m: np.ndarray
    P: np.ndarray


class Bernoulli_GMS_Filter:
    def __init__(self, model, use_gating=True) -> None:
        self.model = model

        self.L_max = 100
        self.elim_threshold = 1e-5
        self.merge_threshold = 4

        self.P_G = 0.999
        self.gamma = chi2.ppf(self.P_G, self.model.z_dim)
        self.use_gating = use_gating

    def init(self):
        r_upds_k = 0.
        w_upds_k = np.array([1.])
        m_upds_k = np.zeros((1, self.model.x_dim))
        P_upds_k = np.eye(self.model.x_dim)[np.newaxis, :]
        return KF_Bernoulli_Data(r_upds_k, w_upds_k, m_upds_k, P_upds_k)

    def predict(self, upds_k):
        N = upds_k.w.shape[0]
        L = self.model.birth_model.Ls[0]

        w_preds_k = np.empty((N+L,))
        m_preds_k = np.empty((N+L, self.model.x_dim))
        P_preds_k = np.empty((N+L, self.model.x_dim, self.model.x_dim))

        r_preds_k = (1 - upds_k.r) * self.model.birth_model.rs \
            + upds_k.r * self.model.survival_model.get_probability()

        # Predict surviving states
        w_preds_k[L:] = upds_k.w * upds_k.r \
            * self.model.survival_model.get_probability()
        m_preds_k[L:], P_preds_k[L:] = \
            kalman_predict(self.model.motion_model.F,
                           self.model.motion_model.Q,
                           upds_k.m, upds_k.P)

        # Predict born states
        w_preds_k[:L] = self.model.birth_model.wss[0] \
            * self.model.birth_model.rs * (1 - upds_k.r)
        m_preds_k[:L] = self.model.birth_model.mss[0]
        P_preds_k[:L] = self.model.birth_model.Pss[0]

        # Normalize prediction
        w_preds_k = w_preds_k / w_preds_k.sum()

        return KF_Bernoulli_Data(r_preds_k, w_preds_k, m_preds_k, P_preds_k)

    def gating(self, Z, preds_k):
        return gate(Z,
                    self.gamma,
                    self.model.measurement_model.H,
                    self.model.measurement_model.R,
                    preds_k.m, preds_k.P)

    def postprocess(self, w_upds_k, m_upds_k, P_upds_k):
        w_upds_k, m_upds_k, P_upds_k = prune(
            w_upds_k, m_upds_k, P_upds_k,
            self.elim_threshold)

        w_upds_k, m_upds_k, P_upds_k = merge_and_cap(
            w_upds_k, m_upds_k, P_upds_k,
            self.merge_threshold, self.L_max)

        return w_upds_k, m_upds_k, P_upds_k

    def update(self, Z, preds_k):
        # == Gating ==
        cand_Z = self.gating(Z, preds_k) \
            if self.use_gating \
            else Z

        # == Update ==
        N1 = preds_k.w.shape[0]
        N2 = cand_Z.shape[0]
        M = N1 * (N2 + 1)

        w_upds_k = np.empty((M,))
        m_upds_k = np.empty((M, self.model.x_dim))
        P_upds_k = np.empty((M, self.model.x_dim, self.model.x_dim))

        # Miss detection
        w_upds_k[:N1] = preds_k.w \
            * (1 - self.model.detection_model.get_probability()) \
            * self.model.clutter_model.lambda_c * self.model.clutter_model.pdf_c
        m_upds_k[:N1] = preds_k.m.copy()
        P_upds_k[:N1] = preds_k.P.copy()

        # Detection
        if N2 > 0:
            qs, ms, Ps = kalman_update(cand_Z,
                                       self.model.measurement_model.H,
                                       self.model.measurement_model.R,
                                       preds_k.m, preds_k.P)

            w_upds_k[N1:] = (self.model.detection_model.get_probability()
                             * preds_k.w * qs.T).reshape(-1)
            m_upds_k[N1:] = ms.transpose(1, 0, 2).reshape(N1 * N2, -1)
            P_upds_k[N1:] = np.tile(Ps, (N2, 1, 1))

        w_ups_k_sum = w_upds_k.sum()
        r_upds_k = preds_k.r * w_ups_k_sum / (
            (1 - preds_k.r)
            * self.model.clutter_model.lambda_c * self.model.clutter_model.pdf_c
            + preds_k.r * w_ups_k_sum
        )

        w_upds_k = w_upds_k / w_ups_k_sum

        # == Post-processing ==
        w_upds_k, m_upds_k, P_upds_k = \
            self.postprocess(w_upds_k, m_upds_k, P_upds_k)

        return KF_Bernoulli_Data(r_upds_k, w_upds_k, m_upds_k, P_upds_k)

    def estimate(self, upds_k):
        idx = [] \
            if upds_k.r <= 0.5 \
            else [np.argmax(upds_k.w)]
        w_ests_k = upds_k.w[idx]
        m_ests_k = upds_k.m[idx]
        P_ests_k = upds_k.P[idx]
        return KF_Bernoulli_Data(upds_k.r, w_ests_k, m_ests_k, P_ests_k)

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

        return [est.w for est in ests],\
            [est.m for est in ests],\
            [est.P for est in ests]
