import numpy as np


class UniformClutterModel:
    def __init__(self, lambda_c, range_c):
        self.lambda_c = lambda_c
        self.range_c = range_c
        self.pdf_c = 1 / np.prod(range_c[:, 1] - range_c[:, 0])

    def sample(self, z_dim):
        N_c = np.random.poisson(self.lambda_c)
        C = self.range_c[:, [0]].T.repeat(N_c, 0) \
            + np.random.rand(N_c, z_dim) \
            @ np.diag(self.range_c[:, 1] - self.range_c[:, 0])
        return C
