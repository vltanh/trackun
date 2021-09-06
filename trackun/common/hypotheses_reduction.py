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
def distance(m1, P1, detP1, iP1, m2, P2, detP2, iP2):
    # Trace term
    # tr = np.trace(iP2 @ P1)

    # Quadratic term
    dx = m1 - m2
    quad = dx.T @ iP2 @ dx

    # Determinant term
    # det = np.log(detP2 / detP1)

    # d = tr + quad + det
    return quad


def merge_components(ws, xs, Ps):
    w_merge = ws.sum()

    x_merge = xs.T @ ws / w_merge

    d = xs - x_merge
    var = d[:, :, np.newaxis] @ d[:, np.newaxis, :]
    P_merge = ((Ps + var).transpose(1, 2, 0) * ws).sum(-1) / w_merge

    return w_merge, x_merge, P_merge


def merge_and_cap(ws, ms, Ps, thres, L_max):
    N, xdim = ms.shape

    indices = ws.argsort()[::-1]
    ws, ms, Ps = ws[indices], ms[indices], Ps[indices]
    detPs = la.det(Ps).reshape(N)
    iPs = la.inv(Ps)

    w_upds = np.empty(L_max)
    m_upds = np.empty((L_max, xdim))
    P_upds = np.empty((L_max, xdim, xdim))

    C = 0
    ignore = np.zeros(N, dtype=np.bool8)
    for i in range(N):
        if ignore[i]:
            continue

        merge_indices = [i]
        merge_indices.extend(filter(
            lambda j:
                not ignore[j] and
                distance(
                    ms[j], Ps[j], detPs[j], iPs[j],
                    ms[i], Ps[i], detPs[i], iPs[i],
                    # ms[j], Ps[j], detPs[j], iPs[j],
                ) < thres,
            range(i+1, N)
        ))

        w_upds[C], m_upds[C], P_upds[C] = \
            merge_components(ws[merge_indices],
                             ms[merge_indices],
                             Ps[merge_indices])
        ignore[merge_indices] = True

        C += 1
        if C >= L_max:
            break

    return w_upds[:C], m_upds[:C], P_upds[:C]
