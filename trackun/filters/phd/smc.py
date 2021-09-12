from dataclasses import dataclass

from trackun.filters.base import SMCFilter
from trackun.common.kmeans import *

import numpy as np

__all__ = ['PHD_SMC_Filter']


@dataclass
class PF_PHD_Data:
    w: np.ndarray
    x: np.ndarray


class PHD_SMC_Filter(SMCFilter):
    def __init__(self, model) -> None:
        self.model = model

        self.J_max = 100000
        self.J_target = 1000
        self.J_birth = model.birth_model.N * self.J_target

    def init(self):
        w = np.zeros(1)
        x = np.zeros((1, self.model.x_dim))
        return PF_PHD_Data(w, x)

    def predict(self, upds_k):
        N = upds_k.w.shape[0]
        L = self.J_birth

        w_preds_k = np.empty((N + L,))
        x_preds_k = np.empty((N + L, self.model.x_dim))

        w_preds_k[L:] = upds_k.w \
            * self.model.survival_model.get_probability(upds_k.x)
        x_preds_k[L:] = \
            self.model.motion_model.get_noisy_next_state(upds_k.x)

        w_preds_k[:L] = np.ones(L) \
            * self.model.birth_model.gm.w.sum() / L
        x_preds_k[:L] = \
            self.model.birth_model.generate_birth_samples(L)

        return PF_PHD_Data(w_preds_k, x_preds_k)

    def update(self, Z, preds_k):
        # == Update ==
        PD_vals = self.model.detection_model.get_probability(preds_k.x)
        rate_c = self.model.clutter_model.rate_c
        pseudo_likelihood = \
            self.model.measurement_model.compute_likelihood(Z,
                                                            preds_k.w, preds_k.x,
                                                            PD_vals, rate_c)

        w_upds_k = pseudo_likelihood * preds_k.w
        x_upds_k = preds_k.x.copy()

        # == Resampling ==
        J_rsp = min(int(w_upds_k.sum() * self.J_target), self.J_max)
        idx = np.random.choice(x_upds_k.shape[0], size=J_rsp,
                               p=w_upds_k/w_upds_k.sum())
        w_upds_k = np.ones(J_rsp) * w_upds_k.sum() / J_rsp
        x_upds_k = x_upds_k[idx].copy()

        return PF_PHD_Data(w_upds_k, x_upds_k)

    def estimate(self, upds_k):
        x_ests_k = [np.empty((0, self.model.x_dim))]
        if upds_k.w.sum() > 0.5:
            x_c, I_c = kmeans(upds_k.w, upds_k.x, 0)

            for j in range(len(x_c)):
                if upds_k.w[I_c[j]].sum() > 0.5:
                    x_ests_k.append(x_c[j])
        x_ests_k = np.vstack(x_ests_k)

        return PF_PHD_Data(upds_k.w, x_ests_k)
