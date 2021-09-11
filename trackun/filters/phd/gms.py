from trackun.common.kalman import kalman_predict, kalman_update
from trackun.common.hypotheses_reduction import prune, merge_and_cap
from trackun.common.gating import gate

import numpy as np
from scipy.stats.distributions import chi2

__all__ = ['PHD_GMS_Filter']


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

    def run(self, Z):
        K = len(Z)

        w_ests, m_ests, P_ests = [], [], []

        w_upds_k = np.array([1.])
        m_upds_k = np.zeros((1, self.model.x_dim))
        P_upds_k = np.eye(self.model.x_dim)[np.newaxis, :]

        for k in range(K):
            # == Predict ==
            N = w_upds_k.shape[0]
            L = self.model.birth_model.N

            w_preds_k = np.empty((N+L,))
            m_preds_k = np.empty((N+L, self.model.x_dim))
            P_preds_k = np.empty((N+L, self.model.x_dim, self.model.x_dim))

            # Predict surviving states
            w_preds_k[L:] = \
                self.model.survival_model.get_probability() * w_upds_k
            m_preds_k[L:], P_preds_k[L:] = \
                kalman_predict(self.model.motion_model.F,
                               self.model.motion_model.Q,
                               m_upds_k, P_upds_k)

            # Predict born states
            w_preds_k[:L] = self.model.birth_model.ws
            m_preds_k[:L] = self.model.birth_model.ms
            P_preds_k[:L] = self.model.birth_model.Ps

            # == Gating ==
            cand_Z = Z[k]
            if self.use_gating:
                cand_Z = gate(Z[k],
                              self.gamma,
                              self.model.measurement_model.H,
                              self.model.measurement_model.R,
                              m_preds_k, P_preds_k)

            # == Update ==
            N1 = w_preds_k.shape[0]
            N2 = cand_Z.shape[0]
            M = N1 * (N2 + 1)

            m_upds_k = np.empty((M, self.model.x_dim))
            P_upds_k = np.empty((M, self.model.x_dim, self.model.x_dim))
            w_upds_k = np.empty((M,))

            # Miss detection
            m_upds_k[:N1] = m_preds_k.copy()
            P_upds_k[:N1] = P_preds_k.copy()
            w_upds_k[:N1] = w_preds_k \
                * (1 - self.model.detection_model.get_probability())

            # Detection
            if N2 > 0:
                qs, ms, Ps = kalman_update(cand_Z,
                                           self.model.measurement_model.H,
                                           self.model.measurement_model.R,
                                           m_preds_k, P_preds_k)

                w = (w_preds_k * qs.T) \
                    * self.model.detection_model.get_probability()
                w = w / (self.model.clutter_model.lambda_c
                         * self.model.clutter_model.pdf_c
                         + w.sum(1)[:, np.newaxis])
                w_upds_k[N1:] = w.reshape(-1)

                m_upds_k[N1:] = \
                    ms.transpose(1, 0, 2).reshape(-1, self.model.x_dim)
                P_upds_k[N1:] = np.tile(Ps, (N2, 1, 1))

            # == Post-processing ==
            w_upds_k, m_upds_k, P_upds_k = prune(
                w_upds_k, m_upds_k, P_upds_k,
                self.elim_threshold)

            w_upds_k, m_upds_k, P_upds_k = merge_and_cap(
                w_upds_k, m_upds_k, P_upds_k,
                self.merge_threshold, self.L_max)

            # == Estimate ==
            cnt = w_upds_k.round().astype(np.int32)
            w_ests_k = w_upds_k.repeat(cnt, axis=0)
            m_ests_k = m_upds_k.repeat(cnt, axis=0)
            P_ests_k = P_upds_k.repeat(cnt, axis=0)

            w_ests.append(w_ests_k)
            m_ests.append(m_ests_k)
            P_ests.append(P_ests_k)

        return w_ests, m_ests, P_ests
