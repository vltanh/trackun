from src.common.kalman import kalman_predict, kalman_update
from src.common.hypotheses_reduction import prune, merge_and_cap
from src.common.gating import gate

import numpy as np
from scipy.stats.distributions import chi2

__all__ = ['CPHD_GMS_Filter']


def esf(Z):
    n_z = len(Z)
    if n_z == 0:
        return np.array([1.])

    F = np.zeros((2, n_z + 1))
    F[0, 1] = F[1, 1] + Z[0]

    i_n, i_nminus = 1, 0
    for n in range(2, n_z + 1):
        F[i_n, 1] = F[i_nminus, 1] + Z[n - 1]
        for k in range(2, n):
            F[i_n, k] = F[i_nminus, k] + Z[n - 1] * F[i_nminus, k - 1]
        F[i_n, n] = Z[n - 1] * F[i_nminus, n - 1]
        i_n, i_nminus = i_nminus, i_n

    F[i_nminus, 0] = 1
    return F[i_nminus]


class CPHD_GMS_Filter:
    def __init__(self, model, use_gating=True) -> None:
        self.model = model

        self.L_max = 100
        self.elim_threshold = 1e-5
        self.merge_threshold = 4

        self.N_max = 20

        self.P_G = 0.99
        self.gamma = chi2.ppf(self.P_G, self.model.z_dim)
        self.use_gating = use_gating

    def run(self, Z):
        K = len(Z)

        w_upds = [np.array([1.])]
        m_upds = [np.zeros((1, self.model.x_dim))]
        P_upds = [np.eye(self.model.x_dim)[np.newaxis, :]]
        c_upds = [np.hstack([np.array([1.]), np.zeros(self.N_max)])]

        w_ests, m_ests, P_ests = [[]], [[]], [[]]

        for k in range(1, K + 1):
            # == Predict ==
            N = w_upds[-1].shape[0]
            L = self.model.L_birth

            w_preds_k = np.empty((N+L,))
            m_preds_k = np.empty((N+L, self.model.x_dim))
            P_preds_k = np.empty((N+L, self.model.x_dim, self.model.x_dim))

            # Predict surviving states
            w_preds_k[L:] = self.model.P_S * w_upds[-1]
            m_preds_k[L:], P_preds_k[L:] = \
                kalman_predict(self.model.F, self.model.Q,
                               m_upds[-1], P_upds[-1])

            # Predict born states
            w_preds_k[:L] = self.model.w_birth
            m_preds_k[:L] = self.model.m_birth
            P_preds_k[:L] = self.model.P_birth

            # Predict cardinality
            # Surviving
            surviving_c_preds = np.zeros(self.N_max + 1)
            for i in range(self.N_max + 1):
                terms = np.zeros(self.N_max + 1)
                for j in range(i, self.N_max + 1):
                    terms[j] = np.exp(
                        np.sum(np.log(np.arange(i + 1, j + 1)))
                        - np.sum(np.log(np.arange(1, j - i + 1)))
                        + j * np.log(self.model.P_S)
                        + (j - i) * np.log(1. - self.model.P_S)
                    ) * c_upds[-1][j]
                surviving_c_preds[i] = terms.sum()

            # Birth
            c_preds_k = np.zeros(self.N_max + 1)
            for n in range(self.N_max + 1):
                terms = np.zeros(self.N_max + 1)
                for j in range(n + 1):
                    terms[j] = np.exp(
                        - self.model.w_birth.sum()
                        + (n - j) * np.log(self.model.w_birth.sum())
                        - np.sum(np.log(np.arange(1, n - j + 1)))
                    ) * surviving_c_preds[j]
                c_preds_k[n] = terms.sum()

            # Normalize
            c_preds_k = c_preds_k / c_preds_k.sum()

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

            m_upds_k = np.empty((M, self.model.x_dim))
            P_upds_k = np.empty((M, self.model.x_dim, self.model.x_dim))
            w_upds_k = np.empty((M,))

            # Compute Kalman update
            if N2 > 0:
                qs, ms, Ps = kalman_update(cand_Z,
                                           self.model.H, self.model.R,
                                           m_preds_k, P_preds_k)

            # Compute symmetric functions
            XI_vals = self.model.P_D * (qs.T @ w_preds_k) / self.model.pdf_c

            esfvals_E = esf(XI_vals)
            esfvals_D = np.zeros((N2, N2))
            for i in range(N2):
                mask = np.ones_like(XI_vals, dtype=np.bool8)
                mask[i] = False
                esfvals_D[:, i] = esf(XI_vals[mask])

            # Compute upsilons
            upsilon0_E = np.zeros(self.N_max + 1)
            upsilon1_E = np.zeros(self.N_max + 1)
            upsilon1_D = np.zeros((self.N_max + 1, N2))

            for n in range(self.N_max + 1):
                terms0_E = np.zeros(min(N2, n) + 1)
                for j in range(min(N2, n) + 1):
                    terms0_E[j] = np.exp(
                        - self.model.lambda_c
                        + (N2 - j) * np.log(self.model.lambda_c)
                        + np.sum(np.log(np.arange(1, n + 1)))
                        - np.sum(np.log(np.arange(1, n - j + 1)))
                        + (n - j) * np.log(1. - self.model.P_D)
                        - j * np.log(np.sum(w_preds_k))
                    ) * esfvals_E[j]
                upsilon0_E[n] = terms0_E.sum()

                terms1_E = np.zeros(min(N2, n) + 1)
                for j in range(min(N2, n) + 1):
                    if n >= j + 1:
                        terms1_E[j] = np.exp(
                            - self.model.lambda_c
                            + (N2 - j) * np.log(self.model.lambda_c)
                            + np.sum(np.log(np.arange(1, n + 1)))
                            - np.sum(np.log(np.arange(1, n - (j + 1) + 1)))
                            + (n - (j + 1)) * np.log(1. - self.model.P_D)
                            - (j + 1) * np.log(np.sum(w_preds_k))
                        ) * esfvals_E[j]
                upsilon1_E[n] = terms1_E.sum()

                if N2 > 0:
                    terms1_D = np.zeros((min(N2 - 1, n) + 1, N2))
                    for i in range(N2):
                        for j in range(min(N2 - 1, n) + 1):
                            if n >= j + 1:
                                terms1_D[j, i] = np.exp(
                                    - self.model.lambda_c
                                    + ((N2 - 1) - j) *
                                    np.log(self.model.lambda_c)
                                    + np.sum(np.log(np.arange(1, n + 1)))
                                    - np.sum(np.log(np.arange(1, n - (j + 1) + 1)))
                                    + (n - (j + 1)) *
                                    np.log(1. - self.model.P_D)
                                    - (j + 1) * np.log(np.sum(w_preds_k))
                                ) * esfvals_D[j, i]
                    upsilon1_D[n, :] = terms1_D.sum(0)

            # Miss detection
            w_upds_k[:N1] = \
                (upsilon1_E @ c_preds_k) / (upsilon0_E @ c_preds_k) \
                * (1 - self.model.P_D) * w_preds_k
            m_upds_k[:N1] = m_preds_k.copy()
            P_upds_k[:N1] = P_preds_k.copy()

            if N2 > 0:
                for i in range(N2):
                    w_upds_k[N1+i*N1:N1+(i+1)*N1] = \
                        (upsilon1_D[:, i] @ c_preds_k) / \
                        (upsilon0_E @ c_preds_k) * \
                        self.model.P_D * w_preds_k * qs[:, i] / \
                        self.model.pdf_c

                m_upds_k[N1:] = \
                    ms.transpose(1, 0, 2).reshape(-1, self.model.x_dim)
                P_upds_k[N1:] = np.tile(Ps, (N2, 1, 1))

            # Update cardinality
            c_upds_k = upsilon0_E * c_preds_k
            c_upds_k = c_upds_k / c_upds_k.sum()

            # == Post-processing ==
            w_upds_k, m_upds_k, P_upds_k = prune(
                w_upds_k, m_upds_k, P_upds_k,
                self.elim_threshold)

            w_upds_k, m_upds_k, P_upds_k = merge_and_cap(
                w_upds_k, m_upds_k, P_upds_k,
                self.merge_threshold, self.L_max)

            # Log
            w_upds.append(w_upds_k)
            m_upds.append(m_upds_k)
            P_upds.append(P_upds_k)
            c_upds.append(c_upds_k)

            # == Estimate ==
            cnt = w_upds_k.round().astype(np.int32)
            w_ests_k = w_upds_k.repeat(cnt, axis=0)
            m_ests_k = m_upds_k.repeat(cnt, axis=0)
            P_ests_k = P_upds_k.repeat(cnt, axis=0)

            cnt = np.argmax(c_upds_k)
            indices = w_upds_k.argsort()[::-1][:cnt]
            w_ests_k, m_ests_k, P_ests_k = \
                w_upds_k[indices], m_upds_k[indices], P_upds_k[indices]

            w_ests.append(w_ests_k)
            m_ests.append(m_ests_k)
            P_ests.append(P_ests_k)

        return w_ests, m_ests, P_ests
