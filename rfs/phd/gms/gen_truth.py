import numpy as np


class Truth:
    def __init__(self, model) -> None:
        self.K = 300
        self.X = [[] for _ in range(self.K)]
        self.N = np.zeros(self.K).astype(int)
        self.L = [[] for _ in range(self.K)]

        self.track_list = [[] for _ in range(self.K)]
        self.total_tracks = 12

        xstart = np.array([
            [0., 0., 0., -10.],
            [400., -10., -600., 5.],
            [-800., 20., -200., -5.],
            [400, -7, -600, -4],
            [400, -2.5, -600, 10],
            [0, 7.5, 0, -5],
            [-800, 12, -200, 7],
            [-200, 15, 800, -10],
            [-800, 3, -200, 15],
            [-200, -3, 800, -15],
            [0, -20, 0, -15],
            [-200, 15, 800, -5],
        ])
        tbirth = np.array([
            1, 1, 1,
            20, 20, 20,
            40, 40,
            60, 60,
            80, 80,
        ])
        tdeath = np.array([
            70, self.K, 70,
            self.K, self.K, self.K,
            self.K, self.K,
            self.K, self.K,
            self.K, self.K,
        ])

        for i, (state, tb, td) in enumerate(zip(xstart, tbirth, tdeath)):
            for k in range(tb, min(td, self.K) + 1):
                state = model.F @ state
                self.X[k-1].append(state)
                self.track_list[k-1].append(i)
                self.N[k-1] += 1

        for k in range(self.K):
            self.X[k] = np.vstack(self.X[k])
            self.track_list[k] = np.array(self.track_list[k])


def gen_truth(model):
    return Truth(model)
