from trackun.models.motion import ConstantVelocityGaussianMotionModel
from trackun.models.measurement import ConstantVelocityGaussianMeasurementModel
from trackun.models.clutter import UniformClutterModel
from trackun.models.birth import MultiBernoulliGaussianBirthModel, MultiBernoulliMixtureGaussianBirthModel
from trackun.models.survival import ConstantSurvivalModel
from trackun.models.detection import ConstantDetectionModel
from trackun.metrics.ospa import OSPA
from examples.generate_data import Observation, Truth
from examples.visualize import draw_ellipse, plot_2d_gaussian_mixture
from examples.constant import COLOR

import numpy as np

__all__ = [
    'LinearGaussianWithBirthModel',
]


class LinearGaussianWithBirthObservation(Observation):
    def get_obs(self, k):
        Zs = self.Z[k]
        X = Zs[:, 0]
        Y = Zs[:, 1]
        return X, Y


class LinearGaussianWithBirthEstimation:
    def __init__(self, est):
        self.gm = est.gm
        if hasattr(est, 'label'):
            self.label = est.label

    def visualize(self, ax, color, label):
        if hasattr(self, 'label'):
            for i in range(len(self.label)):
                ax.scatter(self.gm.m[i, 0], self.gm.m[i, 2],
                           c=[COLOR[self.label[i] % len(COLOR)]], s=10)
                draw_ellipse(self.gm.m[i][[0, 2]], self.gm.P[i][[0, 2]][:, [0, 2]],
                             ax, color=COLOR[self.label[i] % len(COLOR)], fill=None, linestyle='solid')
                ax.arrow(*self.gm.m[i][[0, 2]], *3 *
                         self.gm.m[i][[1, 3]],
                         color=COLOR[self.label[i] % len(COLOR)])

        else:
            plot_2d_gaussian_mixture(self.gm, ax,
                                     color=color, linestyle='solid',
                                     label=label)
            for m in self.gm.m:
                ax.arrow(*m[[0, 2]], *3*m[[1, 3]], color=color)

    def ospa(self, gt):
        ospa = OSPA()
        return ospa(gt, self.gm.m)

    def count(self):
        return len(self.gm.w)


class LinearGaussianWithBirthModel:
    def __init__(self) -> None:
        # Motion model
        self.motion_model = ConstantVelocityGaussianMotionModel(dim=2,
                                                                noise_std=5)

        # Measurement model
        R = np.diag([100., 100.])
        self.measurement_model = ConstantVelocityGaussianMeasurementModel(dim=2,
                                                                          noise_cov=R)

        # Dimensions
        self.x_dim = self.motion_model.x_dim
        self.z_dim = self.measurement_model.z_dim

        # Birth model
        # self.birth_model = MultiBernoulliGaussianBirthModel()
        # P0 = np.diag([10., 10., 10., 10.]) ** 2
        # self.birth_model.add(w=.03, m=[0., 0., 0., 0.],       P=P0)
        # self.birth_model.add(w=.03, m=[400., 0., -600., 0.],  P=P0)
        # self.birth_model.add(w=.03, m=[-800., 0., -200., 0.], P=P0)
        # self.birth_model.add(w=.03, m=[-200., 0., 800., 0.],  P=P0)

        self.birth_model = MultiBernoulliMixtureGaussianBirthModel()
        self.birth_model.add(r=.03,
                             ws=[1.],
                             ms=[[0., 0., 0., 0.]],
                             Ps=[np.diag([10., 10., 10., 10.]) ** 2])
        self.birth_model.add(r=.03,
                             ws=[1.],
                             ms=[[400., 0., -600., 0.]],
                             Ps=[np.diag([10., 10., 10., 10.]) ** 2])
        self.birth_model.add(r=.03,
                             ws=[1.],
                             ms=[[-800., 0., -200., 0.]],
                             Ps=[np.diag([10., 10., 10., 10.]) ** 2])
        self.birth_model.add(r=.03,
                             ws=[1.],
                             ms=[[-200., 0., 800., 0.]],
                             Ps=[np.diag([10., 10., 10., 10.]) ** 2])

        # Survival model
        self.survival_model = ConstantSurvivalModel(.99)

        # Detection paramters
        self.detection_model = ConstantDetectionModel(.98)

        # Clutter model
        range_c = np.array([
            [-1000, 1000],
            [-1000, 1000]
        ])
        self.clutter_model = UniformClutterModel(30, range_c)

        self.unique_labels = dict()
        self.next_id = 0

    def gen_truth(self):
        xstart = np.array([
            [0., 0., 0., -10.],
            [400., -10., -600., 5.],
            [-800., 20., -200., -5.],
            [400, -7, -600, -4],
            [400, -2.5, -600, 10],
            [0, 7.5, 0, -5],
            [-800, 12, -200, 7],
            [-200, 15, 800, -10],
            [-800, 3, -200, 15],
            [-200, -3, 800, -15],
            [0, -20, 0, -15],
            [-200, 15, 800, -5],
        ])[:, np.newaxis, :]
        tbirth = np.array([
            1, 1, 1,
            20, 20, 20,
            40, 40,
            60, 60,
            80, 80,
        ])
        tdeath = np.array([
            70, 100, 70,
            100, 100, 100,
            100, 100,
            100, 100,
            100, 100,
        ])

        truth = Truth(100)
        truth.generate(self, xstart, tbirth, tdeath)
        return truth

    def gen_obs(self, truth):
        return LinearGaussianWithBirthObservation(self, truth)

    def gen_vis_obj(self, est):
        if hasattr(est, 'l'):
            est.label = []
            for i in range(len(est.l)):
                _id = (est.l[i][0], est.l[i][1])
                if _id in self.unique_labels:
                    current_id = self.unique_labels[_id]
                else:
                    current_id = self.next_id
                    self.unique_labels[_id] = self.next_id
                    self.next_id += 1
                est.label.append(current_id)

        return LinearGaussianWithBirthEstimation(est)

    def get_vis_lim(self):
        return (-1000, 1000), (-1000, 1000)
