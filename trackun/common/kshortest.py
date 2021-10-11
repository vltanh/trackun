from dataclasses import dataclass
from typing import List
import numpy as np

np.random.seed(3698)
EPS = np.finfo(np.float64).eps
INF = np.inf


def BFMSpathOT(G, r):
    tail, head = np.where(np.isfinite(G))
    W = G[tail, head]

    _, n = G.shape
    m = len(W)
    p = np.full(n, -1)
    D = np.full(n, np.inf)

    p[r] = -1
    D[r] = 0
    for _ in range(n-1):
        optimal = True
        for arc in range(m):
            u = tail[arc]
            v = head[arc]
            duv = W[arc]
            if D[v] > D[u] + duv:
                D[v] = D[u] + duv
                p[v] = u
                optimal = False
        if optimal:
            break
    return p, D


def BFMSpathwrap(ncm, source, destination):
    p, D = BFMSpathOT(ncm, source)

    pred = p
    dist = D[destination]

    if np.isinf(dist):
        path = []
    else:
        path = [destination]
        while path[-1] != source:
            path.append(pred[path[-1]])
        path = path[::-1]
    return dist, path


@dataclass
class Pcell:
    path: List[int]
    cost: np.ndarray


@dataclass
class Xcell:
    path_number: int
    path: List[int]
    cost: np.ndarray


def find_k_shortest_path(netCostMatrix, source, destination, k_paths):
    if not 0 <= source < netCostMatrix.shape[0] or not 0 < destination <= netCostMatrix.shape[0]:
        print('[WARNING] Source/Destination node is not part of the cost matrix.')
        return [], []

    k = 1
    cost, path = BFMSpathwrap(netCostMatrix, source, destination)

    if len(path) == 0:
        return [], []

    path_number = 0
    P = [Pcell(path, cost)]
    current_P = path_number

    X = [Xcell(path_number, path, cost)]
    size_X = 1

    S = [path[0]]

    shortestPaths = [path.copy()]
    totalCosts = [cost.copy()]

    while k < k_paths and size_X > 0:
        for i in range(len(X)):
            if X[i].path_number == current_P:
                size_X = size_X - 1
                X.pop(i)
                break

        P_ = P[current_P].path

        w = S[current_P]
        for i in range(len(P_)):
            if w == P_[i]:
                w_index_in_path = i

        for index_dev_vertex in range(w_index_in_path, len(P_) - 1):
            temp_netCostMatrix = netCostMatrix.copy()

            for i in range(index_dev_vertex - 1):
                v = P_[i]
                temp_netCostMatrix[v] = np.inf
                temp_netCostMatrix[:, v] = np.inf

            # ===============

            index = 1
            SP_sameSubPath = [P_.copy()]

            for i in range(len(shortestPaths)):
                if len(shortestPaths[i]) >= index_dev_vertex:
                    if P_[:index_dev_vertex + 1] == shortestPaths[i][:index_dev_vertex + 1]:
                        index += 1
                        SP_sameSubPath.append(shortestPaths[i])

            v_ = P_[index_dev_vertex]
            for j in range(len(SP_sameSubPath)):
                _next = SP_sameSubPath[j][index_dev_vertex+1]
                temp_netCostMatrix[v_, _next] = np.inf

            # ==============

            sub_P = P_[:index_dev_vertex+1]
            cost_sub_P = 0
            for i in range(len(sub_P) - 1):
                cost_sub_P += netCostMatrix[sub_P[i], sub_P[i+1]]

            c, dev_p = BFMSpathwrap(temp_netCostMatrix,
                                    P_[index_dev_vertex],
                                    destination)
            if len(dev_p):
                path_number = path_number + 1

                path_t = sub_P[:-1] + dev_p
                cost_t = cost_sub_P + c

                P.append(Pcell(path_t, cost_t))
                S.append(P_[index_dev_vertex])

                size_X += 1
                X.append(Xcell(path_number, path_t, cost_t))

        if size_X > 0:
            shortestXcost = X[0].cost
            shortestX = X[0].path_number
            for i in range(1, size_X):
                if X[i].cost < shortestXcost:
                    shortestX = X[i].path_number
                    shortestXcost = X[i].cost
            current_P = shortestX

            k += 1
            shortestPaths.append(P[current_P].path)
            totalCosts.append(P[current_P].cost)
    return shortestPaths, totalCosts


if __name__ == '__main__':
    C = np.array([
        [INF, 3, 2, INF, INF, INF],
        [INF, INF, INF, 4, INF, INF],
        [INF, 1, INF, 2, 3, INF],
        [INF, INF, INF, INF, 2, 1],
        [INF, INF, INF, INF, INF, 2],
        [INF, INF, INF, INF, INF, INF]
    ])
    paths, cost = find_k_shortest_path(C, 0, 5, 100)
    print(paths, cost)
