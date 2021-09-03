from scipy.spatial import distance_matrix
from scipy.optimize import linear_sum_assignment


def ospa(X, Y, c=100., p=1.):
    nx, ny = X.shape[0], Y.shape[0]

    D = distance_matrix(X, Y)
    D = D.clip(max=c) ** p

    row_ind, col_ind = linear_sum_assignment(D)
    cost = D[row_ind, col_ind].sum()

    ospa_total = (1 / max(nx, ny) * (c ** p * abs(nx - ny) + cost)) ** (1 / p)
    ospa_loc = (1 / max(nx, ny) * cost) ** (1 / p)
    ospa_card = (1 / max(nx, ny) * c ** p * abs(nx - ny)) ** (1 / p)
    return ospa_total, ospa_loc, ospa_card
