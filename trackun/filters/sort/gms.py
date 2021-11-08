from dataclasses import dataclass

import numpy as np
from scipy.spatial import distance_matrix
from scipy.optimize import linear_sum_assignment

from trackun.common.gaussian_mixture import GaussianMixture
from trackun.common.kalman_filter import KalmanFilter
from trackun.filters.base import BayesFilter

__all__ = [
    'SORT',
]


@dataclass
class Track:
    gm: GaussianMixture
    l: np.ndarray


def linear_assignment(cost_matrix):
    x, y = linear_sum_assignment(cost_matrix)
    return np.array(list(zip(x, y)))


class KalmanBoxTracker(object):
    count = 0

    def __init__(self, x):
        # Initialize mean and covariance
        n = 2 * x.shape[-1]

        self.x = np.zeros(n)
        self.x[::2] = x

        self.P = 10. * np.eye(n)
        self.P[::2, ::2] *= 1000.

        # Assign id
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1

        # Log time since the last update
        self.time_since_update = 0

        # Log number of updates
        self.hits = 0
        # Log longest update streak
        self.hit_streak = 0

        # Log time since born
        self.age = 0

    def predict(self, F, Q):
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1

        self.age += 1

        # print(self.x)
        # print(self.P)
        self.x, self.P = \
            KalmanFilter.predict(F, Q,
                                 self.x, self.P)
        # print(self.x)
        # print(self.P)
        # input()

        return self.x

    def update(self, z, H, R):
        self.time_since_update = 0

        self.hits += 1
        self.hit_streak += 1

        _, x, P = \
            KalmanFilter.update(z[np.newaxis],
                                H, R,
                                self.x[np.newaxis], self.P[np.newaxis])
        self.x = x[0, 0]
        self.P = P[0]

        return self.x

    def get_state(self):
        return self.x


def associate_detections_to_trackers(detections, trackers, threshold):
    if len(trackers) == 0:
        return np.empty((0, 2), dtype=np.int64), \
            np.arange(len(detections)), \
            np.empty(0, dtype=np.int64)

    # Compute distance matrix
    dist_mat = distance_matrix(detections, trackers)

    # Find best assignment
    a = (dist_mat > threshold).astype(np.int64)
    if a.sum(1).max() == 1 and a.sum(0).max() == 1:
        matched_indices = np.stack(np.where(a), axis=1)
    else:
        matched_indices = linear_assignment(dist_mat)

    #
    unmatched_detections = list(
        set(range(len(detections))).difference(matched_indices[:, 0])
    )

    unmatched_trackers = list(
        set(range(len(trackers))).difference(matched_indices[:, 1])
    )

    matches = []
    for m in matched_indices:
        if dist_mat[m[0], m[1]] > threshold:
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(1, 2))
    if len(matches) == 0:
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)

    # import matplotlib.pyplot as plt
    # plt.scatter(detections[:, 0], detections[:, 1], color='red')
    # plt.scatter(trackers[:, 0], trackers[:, 1], color='blue')
    # for i, j in matches:
    #     plt.plot([detections[i, 0], trackers[j, 0]],
    #              [detections[i, 1], trackers[j, 1]],
    #              color='orange')
    # plt.show()

    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)


class SORT(BayesFilter):
    def __init__(self,
                 model,
                 max_age=10,
                 min_hits=3,
                 threshold=50):
        self.model = model

        self.max_age = max_age
        self.min_hits = min_hits
        self.threshold = threshold

    def init(self):
        super().init()
        upd = []
        return upd

    def predict(self, upd):
        for t in range(len(upd)):
            upd[t].predict(self.model.motion_model.F,
                           self.model.motion_model.Q)
        return upd

    def update(self, Z, pred):
        trackers = np.empty((len(pred), self.model.motion_model.x_dim // 2))
        for i in range(len(trackers)):
            trackers[i] = pred[i].x[::2]

        matched, unmatched_dets, unmatched_trks = \
            associate_detections_to_trackers(Z, trackers, self.threshold)

        upd = pred

        # update matched trackers with assigned detections
        for m in matched:
            upd[m[1]].update(Z[m[0], :],
                             self.model.measurement_model.H,
                             self.model.measurement_model.R)

        # create and initialise new trackers for unmatched detections
        for i in unmatched_dets:
            trk = KalmanBoxTracker(Z[i, :])
            upd.append(trk)

        for i in range(len(upd) - 1, -1, -1):
            if upd[i].time_since_update > self.max_age:
                upd.pop(i)

        return upd

    def visualizable_estimate(self, upd):
        w = []
        X = []
        P = []

        L = []

        for i in range(len(upd)):
            if upd[i].time_since_update == 0 and (upd[i].hit_streak >= self.min_hits or self.k <= self.min_hits):
                w.append(1)
                X.append(upd[i].x[np.newaxis])
                P.append(upd[i].P[np.newaxis])
                L.append((0, upd[i].id))

        if len(w):
            w = np.hstack(w)
            X = np.vstack(X)
            P = np.vstack(P)
            gm = GaussianMixture(w, X, P)
            L = np.vstack(L)
        else:
            gm = GaussianMixture.get_empty(0, self.model.motion_model.x_dim)
            L = np.empty(0)

        return Track(gm, L)

    def estimate(self, upd):
        X = []
        L = []

        for i in range(len(upd)):
            if upd[i].time_since_update == 0 and (upd[i].hit_streak >= self.min_hits or self.k <= self.min_hits):
                X.append(upd[i].x[np.newaxis])
                L.append((0, upd[i].id))

        if len(X) > 0:
            return np.vstack(X), np.vstack(L)
        return np.empty((0, self.model.motion_model.x_dim)), np.empty((0, 2))
