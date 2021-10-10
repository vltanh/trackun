from dataclasses import dataclass

from trackun.filters.base import GMSFilter
from trackun.common.gaussian_mixture import GaussianMixture
from trackun.common.kalman_filter import KalmanFilter
from trackun.common.gating import EllipsoidallGating

import numpy as np
import numba

__all__ = [
    'CPHD_GMS_Data',
    'CPHD_GMS_Filter',
]


@dataclass
class CPHD_GMS_Data:
    gm: GaussianMixture
    c: np.ndarray


# @numba.jit(nopython=True)
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


def esf_batch(Z):
    nz, dz = Z.shape
    if dz == 0:
        return np.ones((nz, 1))

    F = np.zeros((2, nz, dz + 1))
    F[0, :, 1] = Z[:, 0]

    i_n, i_nminus = 1, 0
    for n in range(2, dz + 1):
        F[i_n, :, 1] = F[i_nminus, :, 1] + Z[:, n - 1]
        for k in range(2, n):
            F[i_n, :, k] = F[i_nminus, :, k] + \
                Z[:, n - 1] * F[i_nminus, :, k - 1]
        F[i_n, :, n] = Z[:, n - 1] * F[i_nminus, :, n - 1]
        i_n, i_nminus = i_nminus, i_n

    F[i_nminus, :, 0] = 1
    return F[i_nminus]


# @numba.jit(nopython=True)
def predict_cardinality(N_max, P_S, sum_w_birth, c_upds_k):
    aux = np.zeros(N_max + 1)
    aux[1:] = np.cumsum(np.log(np.arange(1, N_max + 1)))

    # Surviving
    # surviving_c_preds = np.zeros(N_max + 1)
    # for i in range(N_max + 1):
    #     # terms = np.zeros(N_max + 1)
    #     # for j in range(i, N_max + 1):
    #     #     terms[j] = np.exp(
    #     #         np.log(np.arange(1, j + 1)).sum()
    #     #         # aux[j]
    #     #         - np.log(np.arange(1, i + 1)).sum()
    #     #         # - aux[i]
    #     #         - np.log(np.arange(1, j - i + 1)).sum()
    #     #         # - aux[j-i]
    #     #         + j * np.log(P_S)
    #     #         + (j - i) * np.log(1. - P_S)
    #     #     )
    #     terms = np.zeros(N_max + 1)
    #     terms[i:] = np.exp(
    #         aux[i:] - aux[i] - aux[:N_max - i + 1]
    #         + np.arange(i, N_max + 1) * np.log(P_S)
    #         + np.arange(0, N_max - i + 1) * np.log1p(-P_S)
    #     )
    #     surviving_c_preds[i] = (terms * c_upds_k).sum()
    triu_ones = np.triu(np.ones(N_max + 1))
    ar = np.arange(0, N_max + 1)
    select_indices = ar - ar.reshape(-1, 1)
    terms = np.exp(
        aux
        - aux.reshape(-1, 1)
        - aux.take(select_indices)
        + ar * np.log(P_S)
        + select_indices * np.log1p(-P_S)
    ) * triu_ones * c_upds_k
    surviving_c_preds = terms.sum(1)

    # Birth
    # c_preds_k = np.zeros(N_max + 1)
    # for i in range(N_max + 1):
    #     # terms = np.zeros(N_max + 1)
    #     # for j in range(i + 1):
    #     #     terms[j] = np.exp(
    #     #         - sum_w_birth
    #     #         + (i - j) * log_sum_w_birth
    #     #         - np.sum(np.log(np.arange(1, i - j + 1)))
    #     #         # - aux[i - j]
    #     #     )
    #     terms = np.zeros(N_max + 1)
    #     terms[:i+1] = np.exp(
    #         - sum_w_birth
    #         + np.arange(i, -1, -1) * np.log(sum_w_birth)
    #         - aux[:i+1][::-1]
    #     )
    #     c_preds_k[i] = (terms * surviving_c_preds).sum()
    # tril_ones = np.tril(np.ones(N_max + 1))
    # ar = np.arange(0, N_max + 1)
    # select_indices = ar.reshape(-1, 1) - ar
    terms = np.exp(
        - sum_w_birth
        + select_indices.T * np.log(sum_w_birth)
        - aux.take(select_indices.T)
    ) * triu_ones.T * surviving_c_preds
    c_preds_k = terms.sum(1)

    # Normalize
    c_preds_k = c_preds_k / c_preds_k.sum()

    return c_preds_k


# @numba.jit(nopython=True)
def compute_upsilon1_D_row(N2, n,
                           lambda_c, P_D,
                           log_sum_w_preds_k, esfvals_D,
                           aux):
    N = min(N2 - 1, n)

    # terms1_D = np.zeros((N + 1, N2))
    # for i in range(N2):
    #     for j in range(N + 1):
    #         if n >= j + 1:
    #             terms1_D[j, i] = np.exp(
    #                 - lambda_c
    #                 + ((N2 - 1) - j) * np.log(lambda_c)
    #                 # + np.sum(np.log(np.arange(1, n + 1)))
    #                 + aux[n]
    #                 # - np.sum(np.log(np.arange(1, n - (j + 1) + 1)))
    #                 - aux[n - (j + 1)]
    #                 + (n - (j + 1)) * np.log1p(-P_D)
    #                 - (j + 1) * log_sum_w_preds_k
    #             ) * esfvals_D[j, i]
    terms1_D = np.zeros(N + 1)
    N_ = min(N + 1, n)
    terms1_D[:N_] = np.exp(
        - lambda_c
        + np.arange(N2 - 1, N2 - N_ - 1, -1) * np.log(lambda_c)
        + aux[n]
        - aux[n - N_:n][::-1]
        + np.arange(n - 1, n - N_ - 1, -1) * np.log1p(-P_D)
        - np.arange(1, N_ + 1) * log_sum_w_preds_k
    )
    return (esfvals_D[:N + 1].T * terms1_D).sum(1)


# @numba.jit(nopython=True)
def compute_upsilon1_E_elem(N2, n,
                            lambda_c, P_D,
                            log_sum_w_preds_k, esfvals_E,
                            aux):
    N = min(N2, n)

    # terms1_E = np.zeros(N + 1)
    # for j in range(N + 1):
    #     if n >= j + 1:
    #         terms1_E[j] = np.exp(
    #             - lambda_c
    #             + (N2 - j) * np.log(lambda_c)
    #             + np.sum(np.log(np.arange(1, n + 1)))
    #             # + aux[n]
    #             - np.sum(np.log(np.arange(1, n - (j + 1) + 1)))
    #             # - aux[n - (j + 1)]
    #             + (n - (j + 1)) * np.log1p(-P_D)
    #             - (j + 1) * log_sum_w_preds_k
    #         )
    terms1_E = np.zeros(N + 1)
    N_ = min(N + 1, n)
    terms1_E[:N_] = np.exp(
        - lambda_c
        + np.arange(N2, N2 - N_, -1) * np.log(lambda_c)
        + aux[n]
        - aux[n - N_:n][::-1]
        + np.arange(n - 1, n - N_ - 1, -1) * np.log1p(-P_D)
        - np.arange(1, N_ + 1) * log_sum_w_preds_k
    )
    return (terms1_E * esfvals_E[:N + 1]).sum()


# @numba.jit(nopython=True)
def compute_upsilon0_E_elem(N2, n,
                            lambda_c, P_D,
                            log_sum_w_preds_k, esfvals_E,
                            aux):
    N = min(N2, n)

    # terms0_E = np.zeros(N + 1)
    # for j in range(N + 1):
    #     terms0_E[j] = np.exp(
    #         - lambda_c
    #         + (N2 - j) * np.log(lambda_c)
    #         + np.sum(np.log(np.arange(1, n + 1)))
    #         # + aux[n]
    #         - np.sum(np.log(np.arange(1, n - j + 1)))
    #         # - aux[n - j]
    #         + (n - j) * np.log(1. - P_D)
    #         - j * log_sum_w_preds_k
    #     )
    terms0_E = np.exp(
        - lambda_c
        + np.arange(N2, N2 - N - 1, -1) * np.log(lambda_c)
        + aux[n]
        - aux[n - N:n + 1][::-1]
        + np.arange(n, n - N - 1, -1) * np.log1p(-P_D)
        - np.arange(0, N + 1) * log_sum_w_preds_k
    )

    return (terms0_E * esfvals_E[:N+1]).sum()


# @numba.jit(nopython=True)
def compute_upsilons(N_max, N2, lambda_c, P_D, w_preds_k, esfvals_E, esfvals_D):
    upsilon0_E = np.zeros(N_max + 1)
    upsilon1_E = np.zeros(N_max + 1)
    upsilon1_D = np.zeros((N_max + 1, N2))

    log_sum_w_preds_k = np.log(np.sum(w_preds_k))
    aux = np.zeros(max(N_max, N2) + 1)
    aux[1:] = np.cumsum(np.log(np.arange(1, max(N_max, N2) + 1)))

    for n in range(N_max + 1):
        upsilon0_E[n] = compute_upsilon0_E_elem(N2, n,
                                                lambda_c, P_D,
                                                log_sum_w_preds_k, esfvals_E, aux)

    for n in range(N_max + 1):
        upsilon1_E[n] = compute_upsilon1_E_elem(N2, n,
                                                lambda_c, P_D,
                                                log_sum_w_preds_k, esfvals_E, aux)

    if N2 > 0:
        for n in range(N_max + 1):
            upsilon1_D[n, :] = compute_upsilon1_D_row(N2, n,
                                                      lambda_c, P_D,
                                                      log_sum_w_preds_k, esfvals_D, aux)

    return upsilon0_E, upsilon1_E, upsilon1_D


class CPHD_GMS_Filter(GMSFilter):
    def __init__(self,
                 model,
                 L_max=100,
                 elim_thres=1e-5,
                 merge_threshold=4,
                 use_gating=True,
                 pG=0.999,
                 N_max=20) -> None:
        super().__init__(model,
                         L_max, elim_thres, merge_threshold,
                         use_gating, pG)

        self.N_max = N_max

    def init(self):
        super().init()
        w = np.array([0.])
        m = np.zeros((1, self.model.x_dim))
        P = np.eye(self.model.x_dim)[np.newaxis, :]

        c = np.zeros(1 + self.N_max)
        c[0] = 1.

        gm = GaussianMixture(w, m, P)
        return CPHD_GMS_Data(gm, c)

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

        # Predict cardinality
        c_preds_k = \
            predict_cardinality(
                self.N_max,
                self.model.survival_model.get_probability(),
                self.model.birth_model.get_expected_birth_cnt(),
                upds_k.c)

        return CPHD_GMS_Data(gm_preds_k, c_preds_k)

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

        # Compute Kalman update
        if N2 > 0:
            qs, ms, Ps = KalmanFilter.update(cand_Z,
                                             self.model.measurement_model.H,
                                             self.model.measurement_model.R,
                                             preds_k.gm.m, preds_k.gm.P)

            # Compute symmetric functions
            XI_vals = (qs.T @ preds_k.gm.w) * \
                self.model.detection_model.get_probability() / self.model.clutter_model.pdf_c

            esfvals_E = esf(XI_vals)
            esfvals_D = np.zeros((N2, N2))
            mask = ~np.eye(N2, dtype=np.bool8)
            # for i in range(N2):
            #     esfvals_D[:, i] = esf(XI_vals[mask[i]])
            esfvals_D = esf_batch(
                XI_vals.reshape(1, -1)
                .repeat(N2, axis=0)[mask]
                .reshape(N2, N2 - 1)
            ).T

            # Compute upsilons
            upsilon0_E, upsilon1_E, upsilon1_D = \
                compute_upsilons(self.N_max, N2,
                                 self.model.clutter_model.lambda_c,
                                 self.model.detection_model.get_probability(),
                                 preds_k.gm.w, esfvals_E, esfvals_D)

            # Miss detection
            gm_upds_k.w[:N1] = preds_k.gm.w \
                * (1 - self.model.detection_model.get_probability()) \
                * (upsilon1_E @ preds_k.c) / (upsilon0_E @ preds_k.c)
            gm_upds_k.m[:N1] = preds_k.gm.m.copy()
            gm_upds_k.P[:N1] = preds_k.gm.P.copy()

            # Detection
            w = (qs * preds_k.gm.w[:, np.newaxis]) \
                * self.model.detection_model.get_probability() \
                * (upsilon1_D.T @ preds_k.c) / (upsilon0_E @ preds_k.c) / \
                self.model.clutter_model.pdf_c
            gm_upds_k.w[N1:] = w.T.reshape(-1)

            gm_upds_k.m[N1:] = ms.transpose(1, 0, 2).reshape(N1 * N2, -1)
            gm_upds_k.P[N1:] = np.tile(Ps, (N2, 1, 1))

            # Update cardinality
            c_upds_k = upsilon0_E * preds_k.c
            c_upds_k = c_upds_k / c_upds_k.sum()
        else:
            gm_upds_k.w[:N1] = preds_k.gm.w \
                * (1 - self.model.detection_model.get_probability())
            gm_upds_k.m[:N1] = preds_k.gm.m.copy()
            gm_upds_k.P[:N1] = preds_k.gm.P.copy()

            c_upds_k = np.ones(1 + self.N_max)
            c_upds_k = c_upds_k / c_upds_k.sum()

        # == Post-processing ==
        gm_upds_k = self.postprocess(gm_upds_k)

        return CPHD_GMS_Data(gm_upds_k, c_upds_k)

    def visualizable_estimate(self, upds_k):
        cnt = np.argmax(upds_k.c)
        gm_ests_k = upds_k.gm.cap(cnt)
        return CPHD_GMS_Data(gm_ests_k, upds_k.c)

    def estimate(self, upds_k):
        cnt = np.argmax(upds_k.c)
        return upds_k.gm.cap(cnt).m
