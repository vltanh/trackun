from src.common.kalman import kalman_predict, kalman_update
from src.common.hypotheses_reduction import prune, merge_and_cap
from src.common.gating import gate

import numpy as np
from scipy.stats.distributions import chi2

__all__ = ['PHD_GMS_Filter']


class PHD_GMS_Filter:
    def __init__(self, model, use_gating=True) -> None:
        self.model = model

        self.L_max = 100
        self.elim_threshold = 1e-5
        self.merge_threshold = 4

        self.P_G = 0.99
        self.gamma = chi2.ppf(self.P_G, self.model.z_dim)
        self.use_gating = use_gating

    def run(self, Z):
        K = len(Z)

        # w_preds, m_preds, P_preds = [[]], [[]], [[]]
        w_upds = [np.array([1.])]
        m_upds = [np.zeros((1, self.model.x_dim))]
        P_upds = [np.eye(self.model.x_dim)[np.newaxis, :]]

        for k in range(1, K + 1):
            # == Predict ==
            N = w_upds[-1].shape[0]
            L = self.model.L_birth

            w_preds_k = np.empty((N+L,))
            m_preds_k = np.empty((N+L, self.model.x_dim))
            P_preds_k = np.empty((N+L, self.model.x_dim, self.model.x_dim))

            # Predict surviving states
            w_preds_k[:N] = self.model.P_S * w_upds[-1]
            m_preds_k[:N], P_preds_k[:N] = \
                kalman_predict(self.model.F, self.model.Q,
                               m_upds[-1], P_upds[-1])

            # Predict born states
            w_preds_k[N:] = self.model.w_birth
            m_preds_k[N:] = self.model.m_birth
            P_preds_k[N:] = self.model.P_birth

            # Log (optional)
            # m_preds.append(m_preds_k)
            # P_preds.append(P_preds_k)
            # w_preds.append(w_preds_k)

            # == Gating ==
            cand_Z = Z[k-1]
            if self.use_gating:
                cand_Z = gate(Z[k-1],
                              self.gamma, self.model.H, self.model.R,
                              m_preds_k, P_preds_k)

            # == Update ==
            N1 = w_preds_k.shape[0]
            N2 = cand_Z.shape[0]
            M = N1 * (N2 + 1)

            # Miss detection
            m_upds_k = np.empty((M, self.model.x_dim))
            P_upds_k = np.empty((M, self.model.x_dim, self.model.x_dim))
            w_upds_k = np.empty((M,))

            m_upds_k[:N1] = m_preds_k.copy()
            P_upds_k[:N1] = P_preds_k.copy()
            w_upds_k[:N1] = (1 - self.model.P_D) * w_preds_k

            # Detection
            if len(cand_Z) > 0:
                qs, ms, Ps = kalman_update(cand_Z,
                                           self.model.H, self.model.R,
                                           m_preds_k, P_preds_k)

                P_upds_k[N1:] = np.tile(Ps, (N2, 1, 1))
                m_upds_k[N1:] = ms.transpose(
                    1, 0, 2).reshape(-1, self.model.x_dim)

                w = self.model.P_D * w_preds_k * qs.T
                w = w / (self.model.lambda_c * self.model.pdf_c +
                         w.sum(1)[:, np.newaxis])
                w_upds_k[N1:] = w.reshape(-1)

            # == Post-processing ==
            w_upds_k, m_upds_k, P_upds_k = prune(
                w_upds_k, m_upds_k, P_upds_k,
                self.elim_threshold)

            w_upds_k, m_upds_k, P_upds_k = merge_and_cap(
                w_upds_k, m_upds_k, P_upds_k,
                self.merge_threshold, self.L_max)

            # Log
            m_upds.append(m_upds_k)
            P_upds.append(P_upds_k)
            w_upds.append(w_upds_k)

        return w_upds, m_upds, P_upds
