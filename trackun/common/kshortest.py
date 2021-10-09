from dataclasses import dataclass
from typing import List
import numpy as np

np.random.seed(3698)
EPS = np.finfo(np.float64).eps
INF = np.inf


def find(G):
    head, tail = np.where(np.isfinite(G.T))
    w = G.T[head, tail]
    return tail, head, w


def Initialize(G):
    tail, head, W = find(G)
    _, n = G.shape
    m = len(W)
    p = np.full(n, -1)
    D = np.full(n, np.inf)
    return m, n, p, D, tail, head, W


def BFMSpathOT(G, r):
    m, n, p, D, tail, head, W = Initialize(G)
    p[r] = -1
    D[r] = 0
    for i in range(n-1):
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
            return p, D, i
    return p, D, i


def BFMSpathwrap(ncm, source, destination):
    p, D, _ = BFMSpathOT(ncm, source)
    dist = D[destination]
    pred = p

    if np.isinf(dist):
        path = []
    else:
        path = [destination]
        while path[0] != source:
            path.insert(0, pred[path[0]])
    return dist, path, pred


@dataclass
class Pcell:
    path: List[int]
    cost: np.ndarray


@dataclass
class Xcell:
    path_number: int
    path: List[int]
    cost: np.ndarray


def kShortestPath_any(netCostMatrix, source, destination, k_paths):
    if not 0 <= source < netCostMatrix.shape[0] or not 0 < destination <= netCostMatrix.shape[0]:
        print('[WARNING] Source/Destination node is not part of the cost matrix.')
        return [], []

    k = 1
    cost, path, _ = BFMSpathwrap(netCostMatrix, source, destination)

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

            c, dev_p, _ = BFMSpathwrap(temp_netCostMatrix, P_[
                                       index_dev_vertex], destination)
            if len(dev_p):
                path_number = path_number + 1

                path_t = sub_P[:-1] + dev_p
                cost_t = cost_sub_P + c

                P.append(Pcell(path_t, cost_t))
                S.append(P_[index_dev_vertex])

                size_X += 1
                X.append(Xcell(path_number, path_t, cost_t))
            else:
                pass
                # print('[WARNING] Empty dev_p!')

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
        else:
            pass
    return shortestPaths, totalCosts


def kshortestwrap_pred(rs, k):
    if k == 0:
        return [], []
    ns = len(rs)
    _is = np.argsort(-rs)
    ds = rs[_is]

    CM = np.full((ns, ns), INF)
    for i in range(ns):
        CM[:i, i] = ds[i]

    CMPad = np.full((ns + 2, ns + 2), INF)
    CMPad[0, 1:-1] = ds
    CMPad[0, -1] = 0.
    CMPad[1:-1, -1] = 0.
    CMPad[1:-1, 1:-1] = CM

    paths, costs = kShortestPath_any(CMPad, 0, ns + 1, k)

    for p in range(len(paths)):
        if np.array_equal(paths[p], np.array([1, ns + 2])):
            paths[p] = []
        else:
            paths[p] = [x - 1 for x in paths[p][1:-1]]
            paths[p] = _is[paths[p]].tolist()
    return paths, costs


if __name__ == '__main__':
    rs = 2 * np.random.random(4) - 1
    print(rs)
    paths, costs = kshortestwrap_pred(rs, 100)
    print(paths, costs)

    C = np.array([
        [INF, 3, 2, INF, INF, INF],
        [INF, INF, INF, 4, INF, INF],
        [INF, 1, INF, 2, 3, INF],
        [INF, INF, INF, INF, 2, 1],
        [INF, INF, INF, INF, INF, 2],
        [INF, INF, INF, INF, INF, INF]
    ])
    paths, cost = kShortestPath_any(C, 0, 5, 100)
    print(paths, cost)
