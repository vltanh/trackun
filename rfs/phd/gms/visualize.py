import matplotlib.pyplot as plt


def visualize(w_upds, m_upds, P_upds, meas=None, truth=None):
    for k in range(1, meas.K + 1):
        plt.figure(figsize=(10, 10), dpi=150)

        if meas is not None:
            # plt.scatter(meas.Z[k-1][meas.is_clutter[k-1] == 0, 0],
            #             meas.Z[k-1][meas.is_clutter[k-1] == 0, 1],
            #             c='red', alpha=0.5,
            #             label='Object')
            plt.scatter(meas.Z[k-1][:, 0],
                        meas.Z[k-1][:, 1],
                        c='black', alpha=0.1,
                        label='Detection')

        if truth is not None:
            plt.scatter(truth.X[k-1][:, 0], truth.X[k-1][:, 2],
                        c='blue', alpha=0.5,
                        label='True state')

        plt.scatter([m[0] for w, m in zip(w_upds[k], m_upds[k]) if w < 0.5],
                    [m[2] for w, m in zip(w_upds[k], m_upds[k]) if w < 0.5],
                    s=20, c='orange', alpha=0.5,
                    label='Rejected Prediction')

        plt.scatter([m[0] for w, m in zip(w_upds[k], m_upds[k]) if w >= 0.5],
                    [m[2] for w, m in zip(w_upds[k], m_upds[k]) if w >= 0.5],
                    s=20, c='green',
                    label='Prediction')

        plt.xlim(-1000, 1000)
        plt.ylim(-1000, 1000)
        plt.title(f'Time: {k:3d}')
        plt.legend(loc='upper right')
        plt.tight_layout()
        plt.savefig(f'output/{k:03d}')
        # plt.show()
        plt.close()
