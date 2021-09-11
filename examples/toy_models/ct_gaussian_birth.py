import numpy as np
from numpy.random import multivariate_normal as randmvn
from numpy.random import randn

import numba

__all__ = [
    'CTGaussianWithBirthModel',
]


class CTGaussianWithBirthTruth:
    def __init__(self, model) -> None:
        self.K = 100
        self.X = [[] for _ in range(self.K)]
        self.N = np.zeros(self.K).astype(int)
        self.L = [[] for _ in range(self.K)]

        self.track_list = [[] for _ in range(self.K)]
        self.total_tracks = 10

        w_turn = 2 * np.pi / 180

        xstart = np.array([
            [1000+3.8676, -10, 1500-11.7457, -10, w_turn/8],
            [-250-5.8857,  20, 1000+11.4102, 3, -w_turn/3],
            [-1500-7.3806, 11, 250+6.7993, 10, -w_turn/2],
            [-1500, 43, 250, 0, 0],
            [250-3.8676, 11, 750-11.0747, 5, w_turn/4],
            [-250+7.3806, -12, 1000-6.7993, -12, w_turn/2],
            [1000, 0, 1500, -10, w_turn/4],
            [250, -50, 750, 0, -w_turn/4],
            [1000, -50, 1500, 0, -w_turn/4],
            [250, -40, 750, 25, w_turn/4],
        ])[:, np.newaxis, :]
        tbirth = np.array([
            1,
            10, 10, 10,
            20,
            40, 40, 40,
            60, 60,
        ])
        tdeath = np.array([
            self.K,
            self.K, self.K, 66,
            80,
            self.K, self.K, 80,
            self.K, self.K,
        ])

        for i, (state, tb, td) in enumerate(zip(xstart, tbirth, tdeath)):
            for k in range(tb, min(td, self.K) + 1):
                state = model.gen_new_state(state)
                self.X[k-1].append(state)
                self.track_list[k-1].append(i)
                self.N[k-1] += 1

        for k in range(self.K):
            self.X[k] = np.vstack(self.X[k])
            self.track_list[k] = np.array(self.track_list[k])


class CTGaussianWithBirthObservation:
    def __init__(self, model, truth) -> None:
        self.K = truth.K
        self.Z = [[] for _ in range(self.K)]

        # Visualization
        self.is_detected = [None for _ in range(self.K)]
        self.is_clutter = [None for _ in range(self.K)]

        nz = model.z_dim

        for k in range(self.K):
            # Generate object detections (with miss detections)
            if truth.N[k] > 0:
                mask = np.random.rand(truth.N[k]) \
                    <= model.compute_PD(truth.X[k])
                obs = np.empty((0, model.z_dim))
                X = truth.X[k][mask]  # N, ns
                N = X.shape[0]
                if N > 0:
                    obs = model.gen_noisy_new_obs(X)
                self.Z[k].append(obs)
                self.is_detected[k] = mask

            # Generate clutter detections
            N_c = np.random.poisson(model.lambda_c)
            C = model.range_c[:, [0]].T.repeat(N_c, 0) \
                + np.random.rand(N_c, model.z_dim) \
                @ np.diag(model.range_c[:, 1] - model.range_c[:, 0])
            self.Z[k].append(C)

            self.Z[k] = np.vstack(self.Z[k])
            self.is_clutter[k] = np.zeros(len(self.Z[k]), dtype=np.bool8)
            self.is_clutter[k][-len(C):] = True


class CTGaussianWithBirthModel:
    def __init__(self) -> None:
        # Basic parameters
        self.x_dim = 5  # state dimension
        self.z_dim = 2  # observation dimension
        self.v_dim = 3  # process noise dimension
        self.w_dim = 2  # observation noise dimension

        # Dynamics model
        self.T = 1.

        sigma_vel = 5.
        sigma_turn = np.pi / 180.

        Bt = sigma_vel * np.array([
            [self.T**2/2],
            [self.T]
        ])
        self.B2 = np.block([
            [Bt, np.zeros((2, 2))],
            [np.zeros((2, 1)), Bt, np.zeros((2, 1))],
            [np.zeros((1, 2)), self.T * sigma_turn]
        ])

        self.B = np.eye(self.v_dim)
        self.Q = self.B @ self.B.T

        # Birth parameters
        self.L_birth = 4
        self.w_birth = np.array([.02, .02, .03, .03])
        self.m_birth = np.array([
            [-1500., 0., 250., 0., 0.],
            [-250., 0., 1000., 0., 0.],
            [250., 0., 750., 0., 0.],
            [1000., 0., 1500., 0., 0.]
        ])

        B_birth = np.zeros((self.L_birth, self.x_dim, self.x_dim))
        B_birth[0, :, :] = np.diag([50, 50, 50, 50, 6 * np.pi / 180.])
        B_birth[1, :, :] = np.diag([50, 50, 50, 50, 6 * np.pi / 180.])
        B_birth[2, :, :] = np.diag([50, 50, 50, 50, 6 * np.pi / 180.])
        B_birth[3, :, :] = np.diag([50, 50, 50, 50, 6 * np.pi / 180.])
        self.P_birth = B_birth @ B_birth.transpose((0, 2, 1))

        # Observation model
        self.D = np.diag([2 * np.pi / 180., 10.])
        self.R = self.D @ self.D.T

        # Clutter model
        self.lambda_c = 10
        self.range_c = np.array([
            [-np.pi / 2., np.pi / 2.],
            [0, 2000]
        ])
        self.pdf_c = 1 / np.prod(self.range_c[:, 1] - self.range_c[:, 0])

    def gen_truth(self):
        return CTGaussianWithBirthTruth(self)

    def gen_obs(self, truth):
        return CTGaussianWithBirthObservation(self, truth)

    def gen_new_state(self, X_prev):
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

    def gen_noisy_new_state(self, X_prev):
        X = self.gen_new_state(X_prev)
        V = randn(X_prev.shape[0], self.v_dim) @ self.B
        X = X + V @ self.B2.T
        return X

    def gen_new_obs(self, X):
        Z = np.empty((X.shape[0], self.z_dim))
        P = X[:, [0, 2]]
        Z[:, 0] = np.arctan2(P[:, 0], P[:, 1])
        Z[:, 1] = np.sqrt((P ** 2).sum(-1))
        return Z

    def gen_noisy_new_obs(self, X):
        X = self.gen_new_obs(X)
        W = randn(X.shape[0], self.w_dim) @ self.D
        X = X + W
        return X

    def compute_PD(self, X):
        M = 0.98
        mid = np.zeros(2)
        cov = np.diag([2000., 2000.]) ** 2

        P = X[:, [0, 2]]
        e_sq = ((
            (P - mid)
            @ np.diag(1 / np.diag(np.sqrt(cov)))
        ) ** 2).sum(-1)

        pD = M * np.exp(-e_sq / 2.)
        return pD

    def compute_PS(self, X):
        return 0.99 * np.ones(X.shape[0])

    def compute_likelihood(self, w, X, Z):
        M = len(Z)

        PD_vals = self.compute_PD(X)

        pseudo_likelihood = 1 - PD_vals

        if M > 0:
            N1 = X.shape[0]
            P = np.empty((N1, 2))
            P[:, 0] = X[:, 0]
            P[:, 1] = X[:, 2]

            Phi = np.empty((N1, 2))
            Phi[:, 0] = np.arctan2(P[:, 0], P[:, 1])
            Phi[:, 1] = np.sqrt(np.sum(P ** 2, -1))

            invD = np.diag(1 / np.diag(self.D))
            log_detD = np.log(np.diag(self.D)).sum()

            for i in range(M):
                e_sq = (((Phi - Z[i]) @ invD) ** 2).sum(-1)
                meas_likelihood = np.exp(
                    - e_sq / 2.
                    - np.log(2 * np.pi)
                    + log_detD
                )

                pseudo_likelihood = pseudo_likelihood + PD_vals * meas_likelihood / \
                    (self.lambda_c * self.pdf_c + (PD_vals * meas_likelihood) @ w)

        return pseudo_likelihood

    def generate_birth_samples(self, N):
        X = np.empty((N, self.x_dim))
        comps = np.random.choice(self.L_birth, size=N,
                                 p=self.w_birth / self.w_birth.sum())
        for i in range(self.L_birth):
            mask = comps == i
            X[mask] = randmvn(self.m_birth[i], self.P_birth[i],
                              size=mask.sum())
        return X
