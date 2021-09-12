import numpy as np
from numpy.random import multivariate_normal as randmvn


class BearingGaussianMeasurementModel:
    def __init__(self, z_dim, R) -> None:
        self.R = R

        self.z_dim = z_dim
        self.w_dim = R.shape[0]

    def get_observation(self, X):
        Z = np.empty((X.shape[0], self.z_dim))
        P = X[:, [0, 2]]
        Z[:, 0] = np.arctan2(P[:, 0], P[:, 1])
        Z[:, 1] = np.sqrt((P ** 2).sum(-1))
        return Z

    def get_noisy_observation(self, X):
        X = self.get_observation(X)
        W = randmvn(np.zeros(self.w_dim),
                    self.R,
                    size=X.shape[0])
        X = X + W
        return X

    def compute_likelihood(self, Z, w, X, PD_vals, rate_c):
        M = len(Z)

        pseudo_likelihood = 1 - PD_vals

        if M > 0:
            N1 = X.shape[0]
            P = np.empty((N1, 2))
            P[:, 0] = X[:, 0]
            P[:, 1] = X[:, 2]

            Phi = np.empty((N1, 2))
            Phi[:, 0] = np.arctan2(P[:, 0], P[:, 1])
            Phi[:, 1] = np.sqrt(np.sum(P ** 2, -1))

            sqrtR = np.sqrt(np.diag(self.R))
            invD = np.diag(1 / sqrtR)
            log_detD = np.log(sqrtR).sum()

            for i in range(M):
                e_sq = (((Phi - Z[i]) @ invD) ** 2).sum(-1)
                meas_likelihood = np.exp(
                    - e_sq / 2.
                    - np.log(2 * np.pi)
                    + log_detD
                )

                pseudo_likelihood = pseudo_likelihood + PD_vals * meas_likelihood / \
                    (rate_c + (PD_vals * meas_likelihood) @ w)

        return pseudo_likelihood
