import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse


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


def visualize(w_upds, m_upds, P_upds, model=None, meas=None, truth=None):
    for k in range(1, meas.K + 1):
        fig, ax = plt.subplots(1, 1, figsize=(10, 10), dpi=100)

        if model is not None:
            for m, P in zip(model.m_birth, model.P_birth):
                draw_ellipse(m[[0, 2]], P[[0, 2]][:, [0, 2]], ax,
                             color='orange', fill=None, linestyle='dashed')

        if truth is not None:
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

            marker_cfg = {
                'marker': '.',
                'markersize': 3,
            }
            ax.plot(*xys[:-2],
                    c='blue', alpha=0.5,
                    **marker_cfg,)
            if len(xys) >= 2:
                ax.plot(xys[-2], xys[-1],
                        c='blue', alpha=0.5,
                        **marker_cfg,
                        label='True trajectory')

        if meas is not None:
            # plt.scatter(meas.Z[k-1][meas.is_clutter[k-1] == 0, 0],
            #             meas.Z[k-1][meas.is_clutter[k-1] == 0, 1],
            #             c='red', alpha=0.5,
            #             label='Object')
            ax.scatter(meas.Z[k-1][:, 0],
                       meas.Z[k-1][:, 1],
                       c='black', alpha=0.3,
                       label='Detection')

        # plt.scatter([m[0] for w, m in zip(w_upds[k], m_upds[k]) if w < 0.5],
        #             [m[2] for w, m in zip(w_upds[k], m_upds[k]) if w < 0.5],
        #             c='red', alpha=0.5,
        #             label='Rejected Prediction')

        ax.scatter([m[0] for w, m in zip(w_upds[k], m_upds[k]) if w >= 0.5],
                   [m[2] for w, m in zip(w_upds[k], m_upds[k]) if w >= 0.5],
                   c='green', s=10,
                   label='Prediction')
        for w, m, P in zip(w_upds[k], m_upds[k], P_upds[k]):
            if w >= 0.5:
                draw_ellipse(m[[0, 2]],
                             P[[0, 2]][:, [0, 2]],
                             ax, color='green', fill=None)

                ax.arrow(*m[[0, 2]], *3*m[[1, 3]], color='green')

        ax.set_xlim(-1000, 1000)
        ax.set_ylim(-1000, 1000)
        ax.set_title(f'Time: {k:3d}')
        ax.legend(loc='upper right')

        fig.tight_layout()
        plt.savefig(f'output/{k:03d}')
        # plt.show()
        plt.close(fig)
