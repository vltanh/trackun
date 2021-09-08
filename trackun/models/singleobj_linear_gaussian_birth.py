import numpy as np
from numpy.random import multivariate_normal as mvn

__all__ = [
    'SingleObjectLinearGaussianWithBirthModel',
]


class SingleObjectLinearGaussianWithBirthTruth:
    def __init__(self, model) -> None:
        self.K = 100
        self.X = [[] for _ in range(self.K)]
        self.N = np.zeros(self.K).astype(int)
        self.L = [[] for _ in range(self.K)]

        self.track_list = [[] for _ in range(self.K)]
        self.total_tracks = 1

        xstart = np.array([
            [0., 3., 0., 9.],
        ])
        tbirth = np.array([
            10,
        ])
        tdeath = np.array([
            80,
        ])

        for i, (state, tb, td) in enumerate(zip(xstart, tbirth, tdeath)):
            for k in range(tb, min(td, self.K) + 1):
                state = model.F @ state
                self.X[k-1].append(state)
                self.track_list[k-1].append(i)
                self.N[k-1] += 1

        for k in range(self.K):
            if len(self.X[k]) > 0:
                self.X[k] = np.vstack(self.X[k])
            else:
                self.X[k] = np.empty((0, model.x_dim))
            self.track_list[k] = np.array(self.track_list[k])


class SingleObjectLinearGaussianWithBirthObservation:
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
                mask = np.random.rand(truth.N[k]) <= model.P_D
                obs = np.empty((0, model.z_dim))
                X = truth.X[k][mask]  # N, ns
                N = X.shape[0]
                if N > 0:
                    W = mvn(np.zeros(nz), model.R, size=N)  # N, nz
                    obs = X @ model.H.T + W
                self.Z[k].append(obs)
                self.is_detected[k] = mask

            # Generate clutter detections
            N_c = np.random.poisson(model.lambda_c)
            C = model.range_c[:, [0]].T.repeat(N_c, 0) \
                + np.random.rand(N_c, model.z_dim) \
                @ np.diag(model.range_c[:, 1] - model.range_c[:, 0])
            self.Z[k].append(C)

            if len(self.Z[k]) > 0:
                self.Z[k] = np.vstack(self.Z[k])
            else:
                self.Z[k] = np.empty((0, model.z_dim))
            self.is_clutter[k] = np.zeros(len(self.Z[k]), dtype=np.bool8)
            self.is_clutter[k][-len(C):] = True


class SingleObjectLinearGaussianWithBirthModel:
    def __init__(self) -> None:
        # Basic parameters
        self.x_dim = 4
        self.z_dim = 2

        # Dynamics model
        T = 1.

        A0 = np.array([
            [1, T],
            [0, 1]
        ])
        self.F = np.block([
            [A0, np.zeros((2, 2))],
            [np.zeros((2, 2)), A0]
        ])

        B0 = np.array([
            [T**2/2],
            [T]
        ])
        B = np.block([
            [B0, np.zeros((2, 1))],
            [np.zeros((2, 1)), B0]
        ])
        sigma_v = 5.
        self.Q = sigma_v ** 2 * B @ B.T

        # Survival model
        self.P_S = .99

        # Birth parameters
        self.T_birth = 1
        self.L_birth = np.array([1], dtype=np.int32)
        self.r_birth = np.array([0.01])
        self.w_birth = [np.array([1.])]
        self.m_birth = [np.array([[0., 0., 0., 0.]])]

        B_birth = np.zeros((self.L_birth[0], self.x_dim, self.x_dim))
        B_birth[0, :, :] = np.diag([1000, 10, 1000, 10])
        self.P_birth = [B_birth @ B_birth.transpose((0, 2, 1))]

        # Observation model
        self.H = np.array([
            [1., 0., 0., 0.],
            [0., 0., 1., 0.]
        ])

        D = np.diag([10., 10.])
        self.R = D @ D.T

        # Detection paramters
        self.P_D = .98

        # Clutter model
        self.lambda_c = 60
        self.range_c = np.array([
            [-1000, 1000],
            [-1000, 1000]
        ])
        self.pdf_c = 1 / np.prod(self.range_c[:, 1] - self.range_c[:, 0])

    def gen_truth(self):
        return SingleObjectLinearGaussianWithBirthTruth(self)

    def gen_obs(self, truth):
        return SingleObjectLinearGaussianWithBirthObservation(self, truth)
