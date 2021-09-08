from scipy.spatial import distance_matrix
from scipy.optimize import linear_sum_assignment

__all__ = [
    'OSPA',
]


class OSPA:
    def __init__(self, c=100., p=1.) -> None:
        self.c = c
        self.p = p

    def __call__(self, X, Y):
        nx, ny = X.shape[0], Y.shape[0]
        if nx == 0 and ny == 0:
            return 0., 0., 0.

        D = distance_matrix(X, Y)
        D = D.clip(max=self.c) ** self.p

        row_ind, col_ind = linear_sum_assignment(D)
        cost = D[row_ind, col_ind].sum()

        ospa_total = (
            1 / max(nx, ny)
            * (self.c ** self.p * abs(nx - ny) + cost)
        ) ** (1 / self.p)
        ospa_loc = (
            1 / max(nx, ny)
            * cost
        ) ** (1 / self.p)
        ospa_card = (
            1 / max(nx, ny)
            * self.c ** self.p * abs(nx - ny)
        ) ** (1 / self.p)
        return ospa_total, ospa_loc, ospa_card
