from trackun.models.birth import *

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from tqdm import tqdm

import os

COLOR = ['green', 'purple', 'red']


def prepare_outdir(output_dir):
    if os.path.isdir(output_dir):
        if os.listdir(output_dir):
            print('[WARNING] Non-empty output directory.')
            choice = input('Proceed [Y/n]? ')
            if choice == 'n':
                print('Cancelled.')
                return
    else:
        os.makedirs(output_dir)


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


def plot_2d_gaussian_mixture(gm, ax, color, linestyle, label=None):
    ax.scatter(gm.m[:, 0], gm.m[:, 2],
               c=color, s=10,
               label=label)
    for m, P in zip(gm.m, gm.P):
        draw_ellipse(m[[0, 2]], P[[0, 2]][:, [0, 2]],
                     ax, color=color, fill=None, linestyle=linestyle)


def visualize_model(model, ax):
    if isinstance(model.birth_model, MultiBernoulliMixtureGaussianBirthModel):
        birth_model = model.birth_model
        if len(birth_model.gms):
            plot_2d_gaussian_mixture(birth_model.gms[0], ax,
                                     color='orange', linestyle='dashed',
                                     label='Birth site')
        if len(birth_model.gms) > 1:
            for gm in birth_model.gms[1:]:
                plot_2d_gaussian_mixture(gm, ax,
                                         color='orange', linestyle='dashed')

    if isinstance(model.birth_model, MultiBernoulliGaussianBirthModel):
        birth_model = model.birth_model
        plot_2d_gaussian_mixture(birth_model.gm, ax,
                                 color='orange', linestyle='dashed',
                                 label='Birth site')


def visualize(ests, model=None, obs=None, truth=None,
              out_dir=None):
    if out_dir is not None:
        prepare_outdir(out_dir)

    if truth is not None:
        truth_cnt = []
    cnts = {
        filter_id: []
        for filter_id in ests.keys()
    }
    OSPA_totals, OSPA_locs, OSPA_cards = [
        {
            filter_id: []
            for filter_id in ests.keys()
        }
        for _ in range(3)
    ]

    for k in tqdm(range(obs.K)):
        fig = plt.figure(figsize=(20, 10), dpi=100)
        gs = fig.add_gridspec(4, 2)

        ax_vis = fig.add_subplot(gs[:, 0])
        ax_count = fig.add_subplot(gs[0, 1])
        ax_ospa = fig.add_subplot(gs[1, 1])
        ax_ospa_loc = fig.add_subplot(gs[2, 1])
        ax_ospa_card = fig.add_subplot(gs[3, 1])

        # Visualize model (if given)
        if model is not None:
            visualize_model(model, ax_vis)

        # Visualize truth (if given)
        if truth is not None:
            truth.visualize(k, ax_vis)

            # Graph plot count
            truth_cnt.append(len(truth.X[k]))
            ax_count.plot(range(1, k+2), truth_cnt,
                          label=f'True count', color='blue')

            # Graph metrics
            for (filter_id, est), color in zip(ests.items(), COLOR):
                ospa_total, ospa_loc, ospa_card = est[k].ospa(truth.X[k])
                OSPA_totals[filter_id].append(ospa_total)
                OSPA_locs[filter_id].append(ospa_loc)
                OSPA_cards[filter_id].append(ospa_card)

                ax_ospa.plot(range(1, k+2), OSPA_totals[filter_id],
                             c=color, label=filter_id)
                ax_ospa_loc.plot(range(1, k+2), OSPA_locs[filter_id],
                                 c=color, label=filter_id)
                ax_ospa_card.plot(range(1, k+2), OSPA_cards[filter_id],
                                  c=color, label=filter_id)

        # Visualize observations
        if obs is not None:
            obs.visualize(k, ax_vis)

        # Visualize estimations
        for (filter_id, est), color in zip(ests.items(), COLOR):
            est[k].visualize(ax_vis, color=color,
                             label=f'Prediction ({filter_id})')

            cnts[filter_id].append(est[k].count())
            ax_count.scatter(range(1, k+2), cnts[filter_id],
                             label=filter_id, color=color)

        if model is not None:
            xlim, ylim = model.get_vis_lim()
            ax_vis.set_xlim(xlim)
            ax_vis.set_ylim(ylim)
        ax_vis.legend(loc='lower left')
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
        if out_dir is not None:
            plt.savefig(f'{out_dir}/{k+1:03d}')
        else:
            plt.show()
        plt.close(fig)


def visualize_input(obs, truth=None, out_dir=None):
    if out_dir is not None:
        prepare_outdir(out_dir)

    for k in tqdm(range(obs.K)):
        fig = plt.figure(figsize=(10, 10), dpi=100)
        gs = fig.add_gridspec(1, 1)
        ax_vis = fig.add_subplot(gs[0])

        if truth is not None:
            truth.visualize(k, ax_vis)

        obs.visualize(k, ax_vis)

        ax_vis.set_xlim(-1000, 1000)
        ax_vis.set_ylim(-1000, 1000)
        ax_vis.legend(loc='upper right')

        plt.suptitle(f'Time: {k+1:3d}')
        fig.tight_layout()

        if out_dir is not None:
            plt.savefig(f'{out_dir}/{k+1:03d}')
        else:
            plt.show()
        plt.close(fig)
