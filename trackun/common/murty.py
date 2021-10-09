import numpy as np
from scipy.optimize import linear_sum_assignment
np.random.seed(3698)


def assignmentoptimal(P):
    row_ind, col_ind = linear_sum_assignment(P)
    cost = P[row_ind, col_ind].sum()
    return col_ind, cost


def murty(P0, m):
    S0, C0 = assignmentoptimal(P0)

    nrows, ncols = P0.shape

    if m == 1:
        return S0[np.newaxis], C0

    N = 1000
    answer_list_P = np.zeros((N, nrows, ncols))
    answer_list_S = np.zeros((N, nrows), dtype=np.int32)
    answer_list_C = np.full(N, np.nan)

    answer_list_P[0] = P0.copy()
    answer_list_S[0] = S0
    answer_list_C[0] = C0

    answer_index_next = 1

    assignments = np.zeros((m, nrows), dtype=np.int32)
    costs = np.zeros(m)

    for i in range(m):
        if np.all(np.isnan(answer_list_C)):
            assignments = assignments[:answer_index_next]
            costs = costs[:answer_index_next]
            break

        # Find lowest cost solution
        idx_top = np.nanargmin(answer_list_C[:answer_index_next])
        # print(i, answer_index_next, answer_list_C[:answer_index_next], idx_top)
        # input()

        # Copy current best solution
        assignments[i] = answer_list_S[idx_top]
        costs[i] = answer_list_C[idx_top]
        # print(assignments)
        # print(costs)
        # input()

        # Copy lowest cost problem
        P_now = answer_list_P[idx_top].copy()
        S_now = answer_list_S[idx_top].copy()
        # print(P_now)
        # print(S_now)
        # input()

        # Delete solution
        answer_list_C[idx_top] = np.nan

        for a in range(len(S_now)):
            aw = a
            aj = S_now[a]

            # print(aw, aj)
            # input()

            if aj != -1:
                P_tmp = P_now.copy()
                # print('Before\n', P_tmp)
                if aj < ncols - nrows:
                    P_tmp[aw, aj] = np.inf
                else:
                    P_tmp[aw, ncols - nrows:] = np.inf

                # print('After\n', P_tmp)
                flag = True
                try:
                    S_tmp, C_tmp = assignmentoptimal(P_tmp)
                except ValueError:
                    flag = False
                # print(S_tmp)
                # print(C_tmp)
                # input()

                if flag:
                    if answer_index_next >= len(answer_list_C):
                        answer_list_P = np.vstack(
                            [answer_list_P, np.zeros_like(answer_list_P)])
                        answer_list_S = np.vstack(
                            [answer_list_S, np.zeros_like(answer_list_S)])
                        answer_list_C = np.vstack(
                            [answer_list_C, np.zeros_like(answer_list_C)])
                    answer_list_P[answer_index_next] = P_tmp.copy()
                    answer_list_S[answer_index_next] = S_tmp.copy()
                    answer_list_C[answer_index_next] = C_tmp.copy()
                    answer_index_next += 1

                v_tmp = P_now[aw, aj].copy()
                P_now[aw, :] = np.inf
                P_now[:, aj] = np.inf
                P_now[aw, aj] = v_tmp

    return assignments, costs


def mbest_wrap(P, m):
    if m == 0:
        return [], []

    n1, n2 = P.shape

    # Padding
    P0 = np.zeros((n1, n2 + n1))
    P0[:, :n2] = P

    # Make cost non-negative
    x = P0.min()
    P0 = P0 - x

    # Murty
    assignments, costs = murty(P0, m)

    # Restore correct costs
    costs = costs + x * (assignments > -1).sum(-1)

    # Remove padding
    assignments = assignments[:, :n1]

    # Set clututer
    assignments[assignments >= n2] = -1

    return assignments, costs


if __name__ == '__main__':
    P = np.random.random((3, 4))
    print(P)

    assignments, costs = mbest_wrap(P, 5)
    print(assignments)
    print(costs)

    assignments, costs = mbest_wrap(P, 1)
    print(assignments)
    print(costs)
