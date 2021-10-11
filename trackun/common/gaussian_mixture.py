import numpy as np
import numpy.linalg as la
from numpy.random import multivariate_normal as randmvn


class GaussianMixture:
    def __init__(self, w, m, P) -> None:
        self.w = w
        self.m = m
        self.P = P

    @staticmethod
    def get_empty(size, dim):
        w = np.empty((size,))
        m = np.empty((size, dim))
        P = np.empty((size, dim, dim))
        return GaussianMixture(w, m, P)

    def select(self, indices):
        return GaussianMixture(self.w[indices].copy(),
                               self.m[indices].copy(),
                               self.P[indices].copy())

    def unpack(self):
        return self.w.copy(), self.m.copy(), self.P.copy()

    def add(self, w, m, P):
        ws = np.append(self.w, w)
        ms = np.append(self.m, m, axis=0)
        Ps = np.append(self.P, P, axis=0)
        return GaussianMixture(ws, ms, Ps)

    def sample(self, size):
        X = np.empty((size, self.m.shape[-1]))
        comps = np.random.choice(self.w.shape[0], size=size,
                                 p=self.w / self.w.sum())
        for i in range(self.w.shape[0]):
            mask = comps == i
            X[mask] = randmvn(self.m[i], self.P[i],
                              size=mask.sum())
        return X

    def copy(self):
        return GaussianMixture(self.w.copy(),
                               self.m.copy(),
                               self.P.copy())

    def __repr__(self) -> str:
        return f'''
            w = {self.w},
            m = {self.m},
            P = {self.P}
        '''

    def prune(self, thres):
        return self.select(self.w > thres)

    def cap(self, L_max):
        return self.select(self.w.argsort()[::-1][:L_max])

    def merge_and_cap(self, thres, L_max):
        N, xdim = self.m.shape

        indices = self.w.argsort()[::-1]
        ws, ms, Ps = self.select(indices).unpack()
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

        return GaussianMixture(w_upds[:C], m_upds[:C], P_upds[:C])


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
