import numpy as np
import numpy.linalg as la
from scipy.stats import multivariate_normal as mvn


def kalman_predict(F, Q, ms, Ps):
    m_preds = ms @ F.T
    P_preds = F @ Ps @ F.T + Q
    return m_preds, P_preds


def kalman_update(zs, H, R, ms, Ps):
    Nx, x_dim = ms.shape
    Nz, _ = zs.shape

    z_preds = ms @ H.T
    innov = zs - z_preds[:, None, :]

    Ss = H @ Ps @ H.T + R
    Ks = Ps @ H.T @ la.inv(Ss)

    m_upds = innov @ Ks.transpose(0, 2, 1) + ms[:, None, :]
    P_upds = (np.eye(x_dim) - Ks @ H) @ Ps

    q_upds = np.empty((Nx, Nz))
    for i, (z_pred, S) in enumerate(zip(z_preds, Ss)):
        q_upds[i] = mvn.pdf(zs, z_pred, S).reshape(Nz)

    return q_upds, m_upds, P_upds
