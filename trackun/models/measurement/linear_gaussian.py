import numpy as np
from numpy.random import multivariate_normal as randmvn


class LinearGaussianMeasurementModel:
    def __init__(self, H, R):
        '''
        Linear Measurement Model with Additive Gaussian Noise
        Arguments:
            H: (zdim, xdim)
            R: (zdim, zdim) 
        '''
        assert H.shape[0] == R.shape[0]
        assert R.shape[0] == R.shape[1]

        self.z_dim, self.x_dim = H.shape

        self.H = H
        self.R = R

    def get_observation(self, X):
        '''
        Arguments:
            X: (N, xdim)
        Returns:
            Z: (N, zdim)
        '''
        return X @ self.H.T

    def get_noisy_observation(self, X):
        Z = self.get_observation(X)
        W = randmvn(np.zeros(self.z_dim),
                    self.R,
                    size=X.shape[0])
        return Z + W


class ConstantVelocityGaussianMeasurementModel(LinearGaussianMeasurementModel):
    def __init__(self, dim, noise_cov):
        H = np.eye(2 * dim)[::2]
        super().__init__(H, noise_cov)
