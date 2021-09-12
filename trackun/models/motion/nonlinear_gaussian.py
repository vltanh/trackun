import numpy as np
from numpy.random import multivariate_normal as randmvn


class CoordinatedTurnGaussianMotionModel:
    def __init__(self, x_dim, Q,
                 sigma_vel, sigma_turn,
                 T=1.) -> None:
        Bt = sigma_vel * np.array([
            [T**2/2],
            [T]
        ])
        self.B2 = np.block([
            [Bt, np.zeros((2, 2))],
            [np.zeros((2, 1)), Bt, np.zeros((2, 1))],
            [np.zeros((1, 2)), T * sigma_turn]
        ])

        self.Q = Q

        self.x_dim = x_dim
        self.v_dim = Q.shape[0]
        self.T = T

    def get_next_state(self, X_prev):
        X = np.zeros_like(X_prev)

        L = X_prev.shape[0]
        omega = X_prev[:, 4]
        tol = 1e-10

        sin_omega_T = np.sin(omega * self.T)
        cos_omega_T = np.cos(omega * self.T)

        a = self.T * np.ones(L)
        b = np.zeros(L)

        idx = np.abs(omega) > tol
        a[idx] = sin_omega_T[idx] / omega[idx]
        b[idx] = (1 - cos_omega_T[idx]) / omega[idx]

        X[:, 0] = X_prev[:, 0] + a * X_prev[:, 1] - b * X_prev[:, 3]
        X[:, 1] = cos_omega_T * X_prev[:, 1] - sin_omega_T * X_prev[:, 3]
        X[:, 2] = b * X_prev[:, 1] + X_prev[:, 2] + a * X_prev[:, 3]
        X[:, 3] = sin_omega_T * X_prev[:, 1] + cos_omega_T * X_prev[:, 3]
        X[:, 4] = X_prev[:, 4]

        return X

    def get_noisy_next_state(self, X_prev):
        X = self.get_next_state(X_prev)
        V = randmvn(np.zeros(self.v_dim),
                    self.Q,
                    size=X.shape[0])
        X = X + V @ self.B2.T
        return X
