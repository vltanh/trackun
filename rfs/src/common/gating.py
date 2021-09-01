import numpy as np
import numpy.linalg as la


def gate(zs, gamma, H, R, ms, Ps):
    Ss = H @ Ps @ H.T + R

    z_preds = ms @ H.T
    innovs = zs - z_preds[:, None, :]

    ds = innovs @ la.inv(Ss) @ innovs.transpose(0, 2, 1)
    ds = np.diagonal(ds, axis1=1, axis2=2)

    mask = (ds < gamma).any(0)

    return zs[mask]