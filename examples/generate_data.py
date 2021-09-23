import numpy as np
from numpy.random import rand


class Truth:
    def __init__(self, K):
        self.K = K
        self.X = [[] for _ in range(K)]
        self.N = np.zeros(K).astype(int)
        self.L = [[] for _ in range(K)]

        self.track_list = [[] for _ in range(K)]
        self.total_tracks = 0

    def generate(self, model, x_start, t_birth, t_death):
        self.total_tracks = len(x_start)

        for i, (state, tb, td) in enumerate(zip(x_start, t_birth, t_death)):
            for k in range(tb, min(td, self.K) + 1):
                state = model.motion_model.get_next_state(state)
                self.X[k-1].append(state)
                self.track_list[k-1].append(i)
                self.N[k-1] += 1

        for k in range(self.K):
            if len(self.X[k]) > 0:
                self.X[k] = np.vstack(self.X[k])
            else:
                self.X[k] = np.empty((0, model.x_dim))
            self.track_list[k] = np.array(self.track_list[k])

    def gen_trajectory(self, k):
        xys = []
        for track_idx in range(self.total_tracks):
            x_p = [x[self.track_list[t] == track_idx, 0]
                   for t, x in enumerate(self.X) if t <= k]
            y_p = [x[self.track_list[t] == track_idx, 2]
                   for t, x in enumerate(self.X) if t <= k]

            x_p = [x[0] for x in x_p if len(x) > 0]
            y_p = [y[0] for y in y_p if len(y) > 0]

            xys.append(x_p)
            xys.append(y_p)
        return xys

    def visualize(self, k, ax):
        xys = self.gen_trajectory(k)

        cfg = {
            'c': 'blue',
            'alpha': 0.5,
            'marker': '.',
            'markersize': 3,
        }
        ax.plot(*xys[:-2], **cfg,)
        if len(xys) >= 2:
            ax.plot(xys[-2], xys[-1], **cfg,
                    label='True trajectory')


class Observation:
    def __init__(self, model, truth) -> None:
        self.K = truth.K
        self.Z = [[] for _ in range(self.K)]

        # Visualization
        self.is_detected = [None for _ in range(self.K)]
        self.is_clutter = [None for _ in range(self.K)]

        for k in range(self.K):
            # Generate object detections (with miss detections)
            if truth.N[k] > 0:
                mask = rand(truth.N[k]) \
                    <= model.detection_model.get_probability(truth.X[k])
                obs = np.empty((0, model.z_dim))
                X = truth.X[k][mask]  # N, ns
                N = X.shape[0]
                if N > 0:
                    obs = model.measurement_model.get_noisy_observation(X)
                self.Z[k].append(obs)
                self.is_detected[k] = mask

            # Generate clutter detections
            C = model.clutter_model.sample(model.z_dim)
            self.Z[k].append(C)

            if len(self.Z[k]) > 0:
                self.Z[k] = np.vstack(self.Z[k])
            else:
                self.Z[k] = np.empty((0, model.z_dim))
            self.is_clutter[k] = np.zeros(len(self.Z[k]), dtype=np.bool8)
            self.is_clutter[k][-len(C):] = True

    def visualize(self, k, ax, is_clutter=None):
        X, Y = self.get_obs(k)

        ax.scatter(X, Y,
                   c='black', alpha=0.3,
                   label='Detection')

        if is_clutter is not None:
            ax.scatter(X[is_clutter], Y[is_clutter],
                       c='red', alpha=0.3,
                       label='Clutter')
