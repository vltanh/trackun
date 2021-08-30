import numpy as np
import numpy.linalg as la
from scipy.stats.distributions import chi2
from scipy.stats import multivariate_normal as mvn


def kalman_state_predict(F, Q, ms, Ps):
    m_preds = ms @ F.T
    P_preds = F @ Ps @ F.T + Q
    return m_preds, P_preds


# def kalman_update(z, model, ms, Ps):
#     q_upds, m_upds, P_upds = [], [], []
#     for m, P in zip(ms, Ps):
#         z_pred = model.H @ m
#         S = model.H @ P @ model.H.T + model.R

#         K = P @ model.H.T @ la.inv(S)

#         q_upd = mvn.pdf(z, z_pred, S).reshape(z.shape[0],)
#         m_upd = (z - z_pred) @ K.T + m
#         P_upd = (np.eye(model.x_dim) - K @ model.H) @ P

#         q_upds.append(q_upd)
#         m_upds.append(m_upd)
#         P_upds.append(P_upd)
#     return q_upds, m_upds, P_upds


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


def gate(zs, gamma, H, R, ms, Ps):
    Ss = H @ Ps @ H.T + R

    z_preds = ms @ H.T
    innovs = zs - z_preds[:, None, :]

    ds = innovs @ la.inv(Ss) @ innovs.transpose(0, 2, 1)
    ds = np.diagonal(ds, axis1=1, axis2=2)

    mask = (ds < gamma).any(0)

    return zs[mask]


def prune(ws, ms, Ps, thres):
    mask = ws > thres
    return ws[mask], ms[mask], Ps[mask]


def cap(ws, ms, Ps, L_max):
    indices = ws.argsort()[::-1][:L_max]
    return ws[indices], ms[indices], Ps[indices]


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


class GMSFilter:
    def __init__(self, model, use_gating=True) -> None:
        self.model = model

        self.L_max = 100
        self.elim_threshold = 1e-5
        self.merge_threshold = 4

        self.P_G = 0.99
        self.gamma = chi2.ppf(self.P_G, self.model.z_dim)
        self.use_gating = use_gating

    def run(self, meas):
        # w_preds, m_preds, P_preds = [[]], [[]], [[]]
        w_upds = [np.array([1.])]
        m_upds = [np.zeros((1, self.model.x_dim))]
        P_upds = [np.eye(self.model.x_dim)[np.newaxis, :]]

        for k in range(1, meas.K + 1):
            # == Predict ==
            N = w_upds[-1].shape[0]
            L = self.model.L_birth

            w_preds_k = np.empty((N+L,))
            m_preds_k = np.empty((N+L, self.model.x_dim))
            P_preds_k = np.empty((N+L, self.model.x_dim, self.model.x_dim))

            # Predict surviving states
            w_preds_k[:N] = self.model.P_S * w_upds[-1]
            m_preds_k[:N], P_preds_k[:N] = \
                kalman_state_predict(self.model.F, self.model.Q,
                                     m_upds[-1], P_upds[-1])

            # Predict born states
            w_preds_k[N:] = self.model.w_birth
            m_preds_k[N:] = self.model.m_birth
            P_preds_k[N:] = self.model.P_birth

            # Log (optional)
            # m_preds.append(m_preds_k)
            # P_preds.append(P_preds_k)
            # w_preds.append(w_preds_k)

            # == Gating ==
            cand_Z = meas.Z[k-1]
            if self.use_gating:
                cand_Z = gate(meas.Z[k-1],
                              self.gamma, self.model.H, self.model.R,
                              m_preds_k, P_preds_k)

            # == Update ==
            N1 = w_preds_k.shape[0]
            N2 = cand_Z.shape[0]
            M = N1 * (N2 + 1)

            # Miss detection
            m_upds_k = np.empty((M, self.model.x_dim))
            P_upds_k = np.empty((M, self.model.x_dim, self.model.x_dim))
            w_upds_k = np.empty((M,))

            m_upds_k[:N1] = m_preds_k.copy()
            P_upds_k[:N1] = P_preds_k.copy()
            w_upds_k[:N1] = (1 - self.model.P_D) * w_preds_k

            # Detection
            if len(cand_Z) > 0:
                qs, ms, Ps = kalman_update(cand_Z,
                                           self.model.H, self.model.R,
                                           m_preds_k, P_preds_k)

                P_upds_k[N1:] = np.tile(Ps, (N2, 1, 1))
                m_upds_k[N1:] = ms.transpose(
                    1, 0, 2).reshape(-1, self.model.x_dim)

                w = self.model.P_D * w_preds_k * qs.T
                w = w / (self.model.lambda_c * self.model.pdf_c +
                         w.sum(1)[:, np.newaxis])
                w_upds_k[N1:] = w.reshape(-1)

            # == Post-processing ==
            w_upds_k, m_upds_k, P_upds_k = prune(
                w_upds_k, m_upds_k, P_upds_k, self.elim_threshold)

            w_upds_k, m_upds_k, P_upds_k = merge_and_cap(
                w_upds_k, m_upds_k, P_upds_k,
                self.merge_threshold, self.L_max)

            # Log
            m_upds.append(m_upds_k)
            P_upds.append(P_upds_k)
            w_upds.append(w_upds_k)

        return w_upds, m_upds, P_upds


def run_filter(model, meas):
    filter = GMSFilter(model)
    return filter.run(meas)
