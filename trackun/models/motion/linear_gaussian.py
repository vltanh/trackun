import numpy as np
from numpy.random import multivariate_normal as randmvn


class LinearGaussianMotionModel:
    def __init__(self, F, Q):
        '''
        Linear Motion Model with Additive Gaussian Noise
        Arguments:
            F: (xdim, xdim)
            Q: (xdim, xdim)
        '''
        assert F.shape[1] == F.shape[0]
        assert F.shape[0] == Q.shape[0]
        assert Q.shape[0] == Q.shape[1]

        self.x_dim = F.shape[0]

        self.F = F
        self.Q = Q

    def get_next_state(self, X):
        '''
        Arguments:
            X: (N, xdim)
        Returns:
            X_new: (N, xdim)
        '''
        return X @ self.F.T

    def get_noisy_next_state(self, X):
        '''
        Arguments:
            X: (N, xdim)
        Returns:
            X_new: (N, xdim)
        '''
        X = self.get_next_state(X)
        V = randmvn(np.zeros(self.x_dim),
                    self.Q,
                    size=X.shape[0])
        X = X + V
        return X


class ConstantVelocityGaussianMotionModel(LinearGaussianMotionModel):
    def __init__(self, dim, noise_std, T=1.):
        A0 = np.array([
            [1, T],
            [0, 1]
        ])
        F = np.zeros((2*dim, 2*dim))
        for i in range(dim):
            F[2*i:2*i+2, 2*i:2*i+2] = A0

        B0 = np.array([
            [T**2/2],
            [T]
        ])
        B = np.zeros((2 * dim, dim))
        for i in range(dim):
            B[2*i:2*i+2, i:i+1] = B0
        Q = noise_std ** 2 * (B @ B.T)

        super().__init__(F, Q)
