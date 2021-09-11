from trackun.models.birth import MultiBernoulliGaussianBirthModel, MultiBernoulliMixtureGaussianBirthModel
from trackun.metrics import OSPA

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from tqdm import tqdm

import os

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
    if isinstance(model.birth_model, MultiBernoulliMixtureGaussianBirthModel):
        birth_model = model.birth_model
        for m_births, P_births in zip(birth_model.mss, birth_model.Pss):
            for m, P in zip(m_births, P_births):
                draw_ellipse(m[[0, 2]], P[[0, 2]][:, [0, 2]], ax,
                             color='orange', fill=None, linestyle='dashed')

    if isinstance(model.birth_model, MultiBernoulliGaussianBirthModel):
        birth_model = model.birth_model
        for m, P in zip(birth_model.ms, birth_model.Ps):
            draw_ellipse(m[[0, 2]], P[[0, 2]][:, [0, 2]], ax,
                         color='orange', fill=None, linestyle='dashed')


def visualize_obs(obs, ax, is_clutter=None):
    ax.scatter(obs[:, 0],
               obs[:, 1],
               c='black', alpha=0.3,
               label='Detection')

    if is_clutter is not None:
        ax.scatter(obs[is_clutter, 0],
                   obs[is_clutter, 1],
                   c='red', alpha=0.3,
                   label='Clutter')


def visualize_est(ms, Ps, method, ax, color):
    ax.scatter(ms[:, 0], ms[:, 2],
               c=color, s=10,
               label=f'Prediction ({method})')
    for m, P in zip(ms, Ps):
        draw_ellipse(m[[0, 2]],
                     P[[0, 2]][:, [0, 2]],
                     ax, color=color, fill=None)
        ax.arrow(*m[[0, 2]], *3*m[[1, 3]], color=color)


def visualize(w_ests, m_ests, P_ests,
              methods,
              model=None, obs=None, truth=None,
              output_dir='output'):

    if os.path.isdir(output_dir):
        if os.listdir(output_dir):
            print('[WARNING] Non-empty output directory.')
            choice = input('Proceed [Y/n]? ')
            if choice == 'n':
                print('Cancelled.')
                return
    else:
        os.makedirs(output_dir)

    if truth is not None:
        ospa = OSPA()
        ospa_d = dict()
        for i, method in enumerate(methods):
            ospa_d[method] = {
                'total': np.zeros(obs.K),
                'loc': np.zeros(obs.K),
                'card': np.zeros(obs.K)
            }
            for k in range(obs.K):
                t = ospa(truth.X[k], m_ests[i][k])
                ospa_d[method]['total'][k] = t[0]
                ospa_d[method]['loc'][k] = t[1]
                ospa_d[method]['card'][k] = t[2]

    for k in tqdm(range(obs.K)):
        fig = plt.figure(figsize=(20, 10), dpi=100)
        gs = fig.add_gridspec(4, 2)

        ax_vis = fig.add_subplot(gs[:, 0])
        ax_count = fig.add_subplot(gs[0, 1])
        ax_ospa = fig.add_subplot(gs[1, 1])
        ax_ospa_loc = fig.add_subplot(gs[2, 1])
        ax_ospa_card = fig.add_subplot(gs[3, 1])

        if model is not None:
            visualize_model(model, ax_vis)

        if truth is not None:
            visualize_truth(truth, k+1, ax_vis)
            ax_count.plot(range(1, k+2), [len(x) for x in truth.X[:k+1]],
                          label=f'True count', color='blue')
            for method, color in zip(methods, COLOR):
                ax_ospa.plot(range(1, k+2),
                             ospa_d[method]['total'][:k+1],
                             c=color, label=method)
                ax_ospa_loc.plot(range(1, k+2),
                                 ospa_d[method]['loc'][:k+1],
                                 c=color, label=method)
                ax_ospa_card.plot(range(1, k+2),
                                  ospa_d[method]['card'][:k+1],
                                  c=color, label=method)

        if obs is not None:
            visualize_obs(obs.Z[k], ax_vis)

        for i in range(len(methods)):
            visualize_est(m_ests[i][k], P_ests[i][k],
                          methods[i], ax_vis, color=COLOR[i])
            ax_count.scatter(range(1, k+2), [len(m) for m in m_ests[i][:k+1]],
                             label=methods[i], color=COLOR[i])

        ax_vis.set_xlim(-1000, 1000)
        ax_vis.set_ylim(-1000, 1000)
        ax_vis.legend(loc='upper right')
        ax_vis.set_title('Visualization')

        ax_count.set_xlim(0, obs.K + 1)
        ax_count.set_ylim(0, 15)
        ax_count.legend(loc='upper left')
        ax_count.set_title('Number of objects')

        ax_ospa.set_xlim(0, obs.K + 1)
        ax_ospa.set_ylim(0, 100)
        ax_ospa.legend(loc='upper right')
        ax_ospa.set_title('OSPA')

        ax_ospa_loc.set_xlim(0, obs.K + 1)
        ax_ospa_loc.set_ylim(0, 100)
        ax_ospa_loc.legend(loc='upper right')
        ax_ospa_loc.set_title('OSPA loc')

        ax_ospa_card.set_xlim(0, obs.K + 1)
        ax_ospa_card.set_ylim(0, 100)
        ax_ospa_card.legend(loc='upper right')
        ax_ospa_card.set_title('OSPA card')

        plt.suptitle(f'Time: {k+1:3d}')
        fig.tight_layout()
        plt.savefig(f'{output_dir}/{k+1:03d}')
        # plt.show()
        plt.close(fig)


def visualize_input(obs, truth):
    for k in tqdm(range(obs.K)):
        fig = plt.figure(figsize=(10, 10), dpi=100)
        gs = fig.add_gridspec(1, 1)
        ax_vis = fig.add_subplot(gs[0])

        if truth is not None:
            visualize_truth(truth, k+1, ax_vis)

        if obs is not None:
            visualize_obs(obs.Z[k], ax_vis, obs.is_clutter[k])

        ax_vis.set_xlim(-1000, 1000)
        ax_vis.set_ylim(-1000, 1000)
        ax_vis.legend(loc='upper right')

        plt.suptitle(f'Time: {k+1:3d}')
        fig.tight_layout()
        plt.savefig(f'output/{k+1:03d}')
        # plt.show()
        plt.close(fig)
