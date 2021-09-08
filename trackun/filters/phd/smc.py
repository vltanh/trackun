from trackun.common.kmeans import *
import numpy as np

__all__ = ['PHD_SMC_Filter']


class PHD_SMC_Filter:
    def __init__(self, model) -> None:
        self.model = model

        self.J_max = 100000
        self.J_target = 1000
        self.J_birth = model.L_birth * self.J_target

    def run(self, Z):
        K = len(Z)

        w_upds_k = np.zeros(1)
        x_upds_k = np.zeros((1, self.model.x_dim))

        x_ests = []

        for k in range(K):
            # == Predict ==
            N = w_upds_k.shape[0]
            L = self.J_birth

            w_preds_k = np.empty((N + L,))
            x_preds_k = np.empty((N + L, self.model.x_dim))

            w_preds_k[L:] = self.model.compute_PS(x_upds_k) * w_upds_k
            x_preds_k[L:] = self.model.gen_noisy_new_state(x_upds_k)

            w_preds_k[:L] = np.ones(self.J_birth) \
                * self.model.w_birth.sum() / self.J_birth

            x_preds_k[:L] = self.model.generate_birth_samples(self.J_birth)

            # == Update ==
            pseudo_likelihood = \
                self.model.compute_likelihood(w_preds_k, x_preds_k, Z[k])

            w_upds_k = pseudo_likelihood * w_preds_k
            x_upds_k = x_preds_k.copy()

            # == Resampling ==
            J_rsp = min(int(w_upds_k.sum() * self.J_target), self.J_max)
            idx = np.random.choice(x_upds_k.shape[0], size=J_rsp,
                                   p=w_upds_k/w_upds_k.sum())
            w_upds_k = np.ones(J_rsp) * w_upds_k.sum() / J_rsp
            x_upds_k = x_upds_k[idx].copy()

            # == Estimation ==
            x_ests_k = [np.empty((0, self.model.x_dim))]
            if w_upds_k.sum() > 0.5:
                x_c, I_c = kmeans(w_upds_k, x_upds_k, 0)

                for j in range(len(x_c)):
                    if w_upds_k[I_c[j]].sum() > 0.5:
                        x_ests_k.append(x_c[j])
            x_ests_k = np.vstack(x_ests_k)
            x_ests.append(x_ests_k)

        return x_ests
