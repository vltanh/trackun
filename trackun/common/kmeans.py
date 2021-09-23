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

    x_c = np.array(x_c)
    L = x_c.shape[0]

    t = x_c.copy()
    for _ in range(niters):
        I_idx = [[] for _ in range(L)]
        w_acc = np.zeros(L)
        for i in range(Nx):
            d = np.sum((X[i] - x_c) ** 2, -1)
            idx = d.argsort()

            k = 0
            while w_acc[idx[k]] + w[i] > 1:
                k += 1
                if k >= L:
                    print('[WARNING] Increase number of clusters')
                    L += 1
                    x_c = np.vstack([x_c, X[i]])
                    idx[k] = L-1
                    w_acc = np.append(w_acc, 0)
                    I_idx.append([])
                    break
            w_acc[idx[k]] = w_acc[idx[k]] + w[i]
            I_idx[idx[k]].append(i)
        for el in range(L):
            x_c[el] = X[I_idx[el]].T @ w[I_idx[el]] / w[I_idx[el]].sum()
    print(t - x_c)
    input()

    return x_c, I_idx
