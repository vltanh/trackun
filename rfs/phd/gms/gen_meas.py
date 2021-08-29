import numpy as np
from numpy.random import multivariate_normal as mvn


class Measurement:
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
                obs = []
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

            self.Z[k] = np.vstack(self.Z[k])
            self.is_clutter[k] = np.zeros(len(self.Z[k]))
            self.is_clutter[k][-len(C):] = 1


def gen_meas(model, truth):
    return Measurement(model, truth)
