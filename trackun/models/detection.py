import numpy as np


class ConstantDetectionModel:
    def __init__(self, pD):
        self.pD = pD

    def get_probability(self, _=None):
        return self.pD


class BearingGaussianDetectionModel:
    def __init__(self, maximal, m, P) -> None:
        self.maximal = maximal
        self.m = m
        self.P = P

    def get_probability(self, X):
        P = X[:, [0, 2]]
        e_sq = ((
            (P - self.m)
            @ np.diag(1 / np.diag(np.sqrt(self.P)))
        ) ** 2).sum(-1)

        return self.maximal * np.exp(-e_sq / 2.)
