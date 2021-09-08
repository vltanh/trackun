import numpy as np


def kmeans(w, X, niters):
    x_c, I_idx = [], []
    Nx, xdim = X.shape

    r = np.arange(Nx)
    free = np.ones(Nx, dtype=np.bool8)
    while free.any():
        # Pick a random non-assigned point
        i = np.random.choice(r[free])
        xc_cur = X[i]

        # Calculate distance to other points
        d = ((xc_cur - X[free]) ** 2).sum(-1)
        indices = d.argsort()

        # Find the cut-off for the cluster
        w_cumsum = w[r[free][indices]].cumsum()
        k = w_cumsum.searchsorted(1)

        # Indices of points in the cluster
        I_idx_tmp = r[free][indices[:k]]
        I_idx.append(I_idx_tmp)

        # Cluster center
        x_c_i = X[I_idx_tmp].T @ w[I_idx_tmp] / w[I_idx_tmp].sum()
        x_c.append(x_c_i)

        # Set assigned points to true
        free[I_idx_tmp] = False

    return x_c, I_idx
