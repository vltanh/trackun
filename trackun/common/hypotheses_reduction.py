import numpy as np
import numpy.linalg as la
import numba


def prune(ws, ms, Ps, thres):
    mask = ws > thres
    return ws[mask], ms[mask], Ps[mask]


def cap(ws, ms, Ps, L_max):
    indices = ws.argsort()[::-1][:L_max]
    return ws[indices], ms[indices], Ps[indices]


# @numba.jit(nopython=True)
def close(g1, g2, thres):
    k = g1[1].shape[0]
    iS2 = la.inv(g2[2])
    dx = g1[1] - g2[1]
    tr = np.trace(iS2 @ g1[2])
    quad = dx.T @ iS2 @ dx
    det = np.log(la.det(g2[2]) / la.det(g1[2]))
    d = .5 * (tr + quad - k + det)
    return d < thres


def merge_hyp(hyp, idx):
    ws, xs, Ps = zip(*[hyp[i] for i in idx])
    w_merge = sum(ws)
    x_merge = sum([w * x / w_merge for x, w in zip(xs, ws)])
    P_merge = sum([w * P / w_merge for P, w in zip(Ps, ws)]) +\
        sum([w * (x_merge - x) @ (x_merge - x).T / w_merge for x, w in zip(xs, ws)])
    return (w_merge, x_merge, P_merge)


def merge_and_cap(ws, ms, Ps, thres, L_max):
    hyp = sorted(zip(ws, ms, Ps),
                 key=lambda x: x[0], reverse=True)

    hyp_GSF = []
    ignore = np.zeros(len(hyp), dtype=np.bool8)
    for i in range(len(hyp)):
        if ignore[i]:
            continue
        merge_indices = [i]
        for j in range(i+1, len(hyp)):
            if not ignore[j] and close(hyp[i], hyp[j], thres):
                merge_indices.append(j)

        hyp_GSF.append(merge_hyp(hyp, merge_indices))
        ignore[merge_indices] = True

        if len(hyp_GSF) >= L_max:
            break

    w_upds, m_upds, P_upds = zip(*hyp_GSF)

    w_upds = np.array(w_upds)
    m_upds = np.array(m_upds)
    P_upds = np.array(P_upds)

    return w_upds, m_upds, P_upds
