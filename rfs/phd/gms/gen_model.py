import numpy as np


class Model():
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
        self.L_birth = 4
        self.w_birth = [.03, .03, .03, .03]
        self.m_birth = [
            np.array([0., 0., 0., 0.]),
            np.array([400., 0., -600., 0.]),
            np.array([-800., 0., -200., 0.]),
            np.array([-200., 0., 800., 0.])
        ]

        B_birth = np.zeros((self.L_birth, self.x_dim, self.x_dim))
        B_birth[0, :, :] = np.diag([10, 10, 10, 10])
        B_birth[1, :, :] = np.diag([10, 10, 10, 10])
        B_birth[2, :, :] = np.diag([10, 10, 10, 10])
        B_birth[3, :, :] = np.diag([10, 10, 10, 10])

        self.P_birth = B_birth @ B_birth.transpose((0, 2, 1))
        self.P_birth = [x for x in self.P_birth]

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


def gen_model():
    return Model()
