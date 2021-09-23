from trackun.common.gaussian_mixture import GaussianMixture

import numpy as np

__all__ = [
    'MultiBernoulliGaussianBirthModel',
    'MultiBernoulliMixtureGaussianBirthModel',
]


class MultiBernoulliGaussianBirthModel:
    def __init__(self):
        self.N = 0
        self.gm = None
        self.x_dim = 0

    def add(self, w, m, P):
        self.N += 1

        w = np.array(w)
        m = np.array(m)[np.newaxis]
        P = np.array(P)[np.newaxis]
        self.gm = GaussianMixture(w, m, P) \
            if self.gm is None \
            else self.gm.add(w, m, P)

        self.x_dim = m.shape[-1]

    def generate_birth_samples(self, N):
        return self.gm.sample(N)

    def get_birth_sites(self):
        return self.gm.w, self.gm.m, self.gm.P

    def get_expected_birth_cnt(self):
        return self.gm.w.sum()


class MultiBernoulliMixtureGaussianBirthModel:
    def __init__(self):
        self.N = 0
        self.Ls = np.empty(0).astype(np.int32)
        self.rs = np.empty(0)
        self.gms = []

    def add(self, r, ws, ms, Ps):
        assert len(ws) > 0
        assert sum(ws) == 1

        self.Ls = np.append(self.Ls, len(ws))
        self.rs = np.append(self.rs, r)

        ws = np.array(ws)
        ms = np.array(ms)
        Ps = np.array(Ps)
        self.gms.append(GaussianMixture(ws, ms, Ps))

        self.N += 1

    def add_at_idx(self, i, w, m, P):
        assert 0 <= i < self.N

        self.Ls[i] += 1

        w = np.array(w)
        m = np.array(m)
        P = np.array(P)
        self.gms[i] = self.gms[i].add(w, m, P)

    def get_birth_sites(self):
        w, m, P = zip(*[gm.unpack() for gm in self.gms])
        w = np.hstack(w)
        m = np.vstack(m)
        P = np.vstack(P)

        w = w * self.rs.repeat(self.Ls, axis=0)

        return w, m, P

    def get_expected_birth_cnt(self):
        return self.rs.sum()
