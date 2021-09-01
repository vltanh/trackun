import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

COLOR = ['green', 'purple']


def draw_ellipse(position, covariance, ax=None, **kwargs):
    ax = ax or plt.gca()

    # Convert covariance to principal axes
    if covariance.shape == (2, 2):
        U, s, _ = np.linalg.svd(covariance)
        angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
        width, height = 2 * np.sqrt(s)
    else:
        angle = 0
        width, height = 2 * np.sqrt(covariance)

    # Draw the Ellipse
    ax.add_patch(Ellipse(position,
                         5 * width, 5 * height,
                         angle, **kwargs))


def visualize_truth(truth, k, ax):
    xys = []
    for track_idx in range(truth.total_tracks):
        x_p = [x[truth.track_list[t] == track_idx, 0]
               for t, x in enumerate(truth.X) if t < k]
        y_p = [x[truth.track_list[t] == track_idx, 2]
               for t, x in enumerate(truth.X) if t < k]

        x_p = [x[0] for x in x_p if len(x) > 0]
        y_p = [y[0] for y in y_p if len(y) > 0]

        xys.append(x_p)
        xys.append(y_p)

    cfg = {
        'c': 'blue',
        'alpha': 0.5,
        'marker': '.',
        'markersize': 3,
    }
    ax.plot(*xys[:-2], **cfg,)
    if len(xys) >= 2:
        ax.plot(xys[-2], xys[-1], **cfg,
                label='True trajectory')


def visualize_model(model, ax):
    for m, P in zip(model.m_birth, model.P_birth):
        draw_ellipse(m[[0, 2]], P[[0, 2]][:, [0, 2]], ax,
                     color='orange', fill=None, linestyle='dashed')


def visualize_obs(obs, ax):
    ax.scatter(obs[:, 0],
               obs[:, 1],
               c='black', alpha=0.3,
               label='Detection')


def visualize_est(ms, Ps, method, ax, color):
    ax.scatter(ms[:, 0], ms[:, 2],
               c=color, s=10,
               label=f'Prediction ({method})')
    for m, P in zip(ms, Ps):
        draw_ellipse(m[[0, 2]],
                     P[[0, 2]][:, [0, 2]],
                     ax, color=color, fill=None)
        ax.arrow(*m[[0, 2]], *3*m[[1, 3]], color=color)


def graph_truth_count(cnts, ax, color):
    ax.plot(range(len(cnts)), cnts,
            label=f'True count', color=color)


def graph_count(cnts, method, ax, color):
    ax.scatter(range(len(cnts)), cnts,
               label=f'{method}', color=color)


def visualize(w_upds, m_upds, P_upds, methods, model=None, obs=None, truth=None):
    for k in range(1, obs.K + 1):
        fig, ax = plt.subplots(1, 2, figsize=(20, 10), dpi=100)

        if model is not None:
            visualize_model(model, ax[0])
            graph_truth_count([len(x) for x in truth.X[:k]], ax[1], 'blue')

        if truth is not None:
            visualize_truth(truth, k, ax[0])

        if obs is not None:
            visualize_obs(obs.Z[k-1], ax[0])

        for i in range(len(w_upds)):
            visualize_est(m_upds[i][k], P_upds[i][k],
                          methods[i], ax[0], color=COLOR[i])
            graph_count([len(w) for w in w_upds[i][:k]],
                        methods[i], ax[1], COLOR[i])

        ax[0].set_xlim(-1000, 1000)
        ax[0].set_ylim(-1000, 1000)
        ax[0].legend(loc='upper right')
        ax[0].set_title('Visualization')

        ax[1].set_xlim(0, obs.K + 1)
        ax[1].set_ylim(0, 15)
        ax[1].legend(loc='upper right')
        ax[1].set_title('Number of objects')

        plt.suptitle(f'Time: {k:3d}')
        fig.tight_layout()
        plt.savefig(f'output/{k:03d}')
        # plt.show()
        plt.close(fig)
