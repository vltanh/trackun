import numpy as np


class MultiBernoulliGaussianBirthModel:
    def __init__(self):
        self.N = 0
        self.ws = None
        self.ms = None
        self.Ps = None

    def add(self, w, m, P):
        if self.N == 0:
            self.ws = np.array(w)
            self.ms = np.array(m)[np.newaxis]
            self.Ps = np.array(P)[np.newaxis]
        else:
            self.ws = np.append(self.ws, w)
            self.ms = np.append(self.ms,
                                np.array(m)[np.newaxis],
                                axis=0)
            self.Ps = np.append(self.Ps,
                                np.array(P)[np.newaxis],
                                axis=0)
        self.N += 1


class MultiBernoulliMixtureGaussianBirthModel:
    def __init__(self):
        self.N = 0
        self.Ls = np.empty(0).astype(int)
        self.rs = np.empty(0)
        self.wss = []
        self.mss = []
        self.Pss = []

    def add(self, r, ws, ms, Ps):
        self.N += 1
        self.Ls = np.append(self.Ls, len(ws))
        self.rs = np.append(self.rs, r)
        self.wss.append(np.array(ws))
        self.mss.append(np.array(ms))
        self.Pss.append(np.array(Ps))
