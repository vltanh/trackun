import numpy as np
import numpy.linalg as la
from scipy.stats.distributions import chi2
from scipy.stats import multivariate_normal as mvn


def kalman_state_predict(F, Q, ms, Ps):
    m_preds, P_preds = [], []
    for m, P in zip(ms, Ps):
        m_preds.append(F @ m)
        P_preds.append(F @ P @ F.T + Q)
    return m_preds, P_preds


def kalman_update(z, model, ms, Ps):
    q_upds, m_upds, P_upds = [], [], []
    for m, P in zip(ms, Ps):
        z_pred = model.H @ m
        S = model.H @ P @ model.H.T + model.R

        # V = la.cholesky(S)
        # detS = np.prod(np.diag(V)) ** 2
        # isqrS = la.inv(V)
        # iS = isqrS @ isqrS.T

        K = P @ model.H.T @ la.inv(S)

        q_upd = mvn.pdf(z, z_pred, S).reshape(z.shape[0],)
        m_upd = (z - z_pred) @ K.T + m
        P_upd = (np.eye(model.x_dim) - K @ model.H) @ P

        q_upds.append(q_upd)
        m_upds.append(m_upd)
        P_upds.append(P_upd)
    return q_upds, m_upds, P_upds


def gate(z, gamma, model, ms, Ps):
    z_gate = []
    for m, P in zip(ms, Ps):
        S = model.H @ P @ model.H.T + model.R
        innov = z - model.H @ m

        # V = la.cholesky(S)
        # inv_sqrt_S = la.inv(V)
        # d2_ = np.sum((inv_sqrt_S.T @ innov.T) ** 2, 0)
        d2 = np.diag(innov @ la.inv(S) @ innov.T)

        z_gate.append(z[d2 < gamma])
    return np.vstack(z_gate)


def prune(ws, ms, Ps, thres):
    # mask = ws > thres
    # return ws[mask], ms[mask], Ps[mask]
    mr = [m for m, w in zip(ms, ws) if w > thres]
    Pr = [P for P, w in zip(Ps, ws) if w > thres]
    wr = [w for w in ws if w > thres]
    return wr, mr, Pr


def cap(ws, ms, Ps, L_max):
    hyp = zip(ws, ms, Ps)
    sorted_hyp = sorted(hyp, key=lambda x: x[0], reverse=True)
    capped_hyp = sorted_hyp[:L_max]
    return list(zip(*capped_hyp))


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
    P_merge = sum([w * P / w_merge for P, w in zip(Ps, ws)]) + \
        sum([w * (x_merge - x) @ (x_merge - x).T / w_merge for x, w in zip(xs, ws)])
    return (w_merge, x_merge, P_merge)


def merge(ws, ms, Ps, thres):
    hyp = sorted(zip(ws, ms, Ps),
                 key=lambda x: x[0], reverse=True)
    hyp_GSF = []
    ignore = np.zeros(len(hyp), dtype=bool)
    for i in range(len(hyp)):
        if ignore[i]:
            continue
        merge_indices = [i]
        for j in range(i+1, len(hyp)):
            if not ignore[j] and close(hyp[i], hyp[j], thres):
                merge_indices.append(j)
        hyp_GSF.append(merge_hyp(hyp, merge_indices))
        ignore[merge_indices] = True

        # if len(hyp_GSF) >= self.L_max:
        #     break

    return zip(*hyp_GSF)


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
        m_preds, P_preds, w_preds = [[]], [[]], [[]]
        m_upds = [[np.zeros(self.model.x_dim)]]
        P_upds = [[np.eye(self.model.x_dim)]]
        w_upds = [[1.]]

        for k in range(1, meas.K + 1):
            # == Predict ==
            # Predict surviving states
            m_preds_k, P_preds_k = kalman_state_predict(self.model.F, self.model.Q,
                                                        m_upds[-1], P_upds[-1])
            w_preds_k = [
                self.model.P_S * w for w in w_upds[-1]
            ]

            # Predict new states
            m_preds_k.extend(self.model.m_birth)
            P_preds_k.extend(self.model.P_birth)
            w_preds_k.extend(self.model.w_birth)

            # Log (optional)
            m_preds.append(m_preds_k)
            P_preds.append(P_preds_k)
            w_preds.append(w_preds_k)

            # == Gating ==
            cand_Z = meas.Z[k-1]
            if self.use_gating:
                cand_Z = gate(meas.Z[k-1],
                              self.gamma, self.model,
                              m_preds_k, P_preds_k)

            # == Update ==
            # Miss detection
            m_upds_k = [m for m in m_preds_k]
            P_upds_k = [P for P in P_preds_k]
            w_upds_k = [(1 - self.model.P_D) * x for x in w_preds_k]

            # Detection
            if len(cand_Z) > 0:
                qs, ms, Ps = kalman_update(cand_Z, self.model,
                                           m_preds_k, P_preds_k)
                for i in range(len(cand_Z)):
                    w = []
                    for j in range(len(ms)):
                        m_upds_k.append(ms[j][i])
                        P_upds_k.append(Ps[j])

                        w.append(self.model.P_D *
                                 w_preds_k[j] * qs[j][i])
                    w = [
                        ww / (self.model.lambda_c * self.model.pdf_c + sum(w))
                        for ww in w
                    ]
                    w_upds_k.extend(w)

            # m_upds_k = np.array(m_upds_k)
            # P_upds_k = np.array(P_upds_k)
            # w_upds_k = np.array(w_upds_k)

            # == Post-processing ==
            w_upds_k, m_upds_k, P_upds_k = prune(
                w_upds_k, m_upds_k, P_upds_k, self.elim_threshold)

            w_upds_k, m_upds_k, P_upds_k = merge(
                w_upds_k, m_upds_k, P_upds_k, self.merge_threshold)

            w_upds_k, m_upds_k, P_upds_k = cap(
                w_upds_k, m_upds_k, P_upds_k, self.L_max)

            # Log
            m_upds.append(m_upds_k)
            P_upds.append(P_upds_k)
            w_upds.append(w_upds_k)

        return w_upds, m_upds, P_upds


def run_filter(model, meas):
    filter = GMSFilter(model)
    return filter.run(meas)
