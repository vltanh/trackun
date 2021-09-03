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
def close(m1, P1, detP1, iP1, m2, P2, detP2, iP2, thres):
    # Trace term
    tr = np.trace(iP2 @ P1)

    # Quadratic term
    dx = m1 - m2
    quad = dx.T @ iP2 @ dx

    # Determinant term
    det = np.log(detP2 / detP1)

    # Distance
    d = tr + quad + det
    return d < thres


def merge_components(ws, xs, Ps):
    w_merge = sum(ws)
    x_merge = sum([x * w / w_merge for w, x in zip(ws, xs)])
    P_merge = sum([(P + (x_merge - x) @ (x_merge - x).T) *
                  w / w_merge for w, x, P in zip(ws, xs, Ps)])
    return (w_merge, x_merge, P_merge)


def merge_and_cap(ws, ms, Ps, thres, L_max):
    detPs = la.det(Ps).reshape(Ps.shape[0])
    iPs = la.inv(Ps)
    hyp = sorted(zip(ws, ms, Ps, detPs, iPs),
                 key=lambda x: x[0], reverse=True)
    N = len(hyp)

    hyp_GSF = []
    ignore = np.zeros(N, dtype=np.bool8)
    for i in range(N):
        if ignore[i]:
            continue

        merge_indices = [i]
        merge_indices.extend(filter(
            lambda j:
                not ignore[j] and
                close(*hyp[i][1:], *hyp[j][1:], thres),
            range(i+1, N)
        ))

        ws, ms, Ps = zip(*[hyp[i][:3] for i in merge_indices])
        hyp_GSF.append(merge_components(ws, ms, Ps))
        ignore[merge_indices] = True

        if len(hyp_GSF) >= L_max:
            break

    w_upds, m_upds, P_upds = zip(*hyp_GSF)

    w_upds = np.array(w_upds)
    m_upds = np.array(m_upds)
    P_upds = np.array(P_upds)

    return w_upds, m_upds, P_upds
