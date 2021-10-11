from trackun.models.motion import ConstantVelocityGaussianMotionModel
from trackun.models.measurement import ConstantVelocityGaussianMeasurementModel
from trackun.models.clutter import UniformClutterModel
from trackun.models.birth import MultiBernoulliGaussianBirthModel, MultiBernoulliMixtureGaussianBirthModel
from trackun.models.survival import ConstantSurvivalModel
from trackun.models.detection import ConstantDetectionModel
from trackun.metrics.ospa import OSPA
from examples.generate_data import Observation, Truth
from examples.visualize import draw_ellipse, plot_2d_gaussian_mixture

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


COLOR = np.array([
    [0,   0,   0],
    [128,   0,   0],
    [0, 128,   0],
    [128, 128,   0],
    [0,   0, 128],
    [128,   0, 128],
    [0, 128, 128],
    [128, 128, 128],
    [64,   0,   0],
    [192,   0,   0],
    [64, 128,   0],
    [192, 128,   0],
    [64,   0, 128],
    [192,   0, 128],
    [64, 128, 128],
    [192, 128, 128],
    [0,  64,   0],
    [128,  64,   0],
    [0, 192,   0],
    [128, 192,   0],
    [0,  64, 128],
    [128,  64, 128],
    [0, 192, 128],
    [128, 192, 128],
    [64,  64,   0],
    [192,  64,   0],
    [64, 192,   0],
    [192, 192,   0],
    [64,  64, 128],
    [192,  64, 128],
    [64, 192, 128],
    [192, 192, 128],
    [0,   0,  64],
    [128,   0,  64],
    [0, 128,  64],
    [128, 128,  64],
    [0,   0, 192],
    [128,   0, 192],
    [0, 128, 192],
    [128, 128, 192],
    [64,   0,  64],
    [192,   0,  64],
    [64, 128,  64],
    [192, 128,  64],
    [64,   0, 192],
    [192,   0, 192],
    [64, 128, 192],
    [192, 128, 192],
    [0,  64,  64],
    [128,  64,  64],
    [0, 192,  64],
    [128, 192,  64],
    [0,  64, 192],
    [128,  64, 192],
    [0, 192, 192],
    [128, 192, 192],
    [64,  64,  64],
    [192,  64,  64],
    [64, 192,  64],
    [192, 192,  64],
    [64,  64, 192],
    [192,  64, 192],
    [64, 192, 192],
    [192, 192, 192],
    [32,   0,   0],
    [160,   0,   0],
    [32, 128,   0],
    [160, 128,   0],
    [32,   0, 128],
    [160,   0, 128],
    [32, 128, 128],
    [160, 128, 128],
    [96,   0,   0],
    [224,   0,   0],
    [96, 128,   0],
    [224, 128,   0],
    [96,   0, 128],
    [224,   0, 128],
    [96, 128, 128],
    [224, 128, 128],
    [32,  64,   0],
    [160,  64,   0],
    [32, 192,   0],
    [160, 192,   0],
    [32,  64, 128],
    [160,  64, 128],
    [32, 192, 128],
    [160, 192, 128],
    [96,  64,   0],
    [224,  64,   0],
    [96, 192,   0],
    [224, 192,   0],
    [96,  64, 128],
    [224,  64, 128],
    [96, 192, 128],
    [224, 192, 128],
    [32,   0,  64],
    [160,   0,  64],
    [32, 128,  64],
    [160, 128,  64],
    [32,   0, 192],
    [160,   0, 192],
    [32, 128, 192],
    [160, 128, 192],
    [96,   0,  64],
    [224,   0,  64],
    [96, 128,  64],
    [224, 128,  64],
    [96,   0, 192],
    [224,   0, 192],
    [96, 128, 192],
    [224, 128, 192],
    [32,  64,  64],
    [160,  64,  64],
    [32, 192,  64],
    [160, 192,  64],
    [32,  64, 192],
    [160,  64, 192],
    [32, 192, 192],
    [160, 192, 192],
    [96,  64,  64],
    [224,  64,  64],
    [96, 192,  64],
    [224, 192,  64],
    [96,  64, 192],
    [224,  64, 192],
    [96, 192, 192],
    [224, 192, 192],
    [0,  32,   0],
    [128,  32,   0],
    [0, 160,   0],
    [128, 160,   0],
    [0,  32, 128],
    [128,  32, 128],
    [0, 160, 128],
    [128, 160, 128],
    [64,  32,   0],
    [192,  32,   0],
    [64, 160,   0],
    [192, 160,   0],
    [64,  32, 128],
    [192,  32, 128],
    [64, 160, 128],
    [192, 160, 128],
    [0,  96,   0],
    [128,  96,   0],
    [0, 224,   0],
    [128, 224,   0],
    [0,  96, 128],
    [128,  96, 128],
    [0, 224, 128],
    [128, 224, 128],
    [64,  96,   0],
    [192,  96,   0],
    [64, 224,   0],
    [192, 224,   0],
    [64,  96, 128],
    [192,  96, 128],
    [64, 224, 128],
    [192, 224, 128],
    [0,  32,  64],
    [128,  32,  64],
    [0, 160,  64],
    [128, 160,  64],
    [0,  32, 192],
    [128,  32, 192],
    [0, 160, 192],
    [128, 160, 192],
    [64,  32,  64],
    [192,  32,  64],
    [64, 160,  64],
    [192, 160,  64],
    [64,  32, 192],
    [192,  32, 192],
    [64, 160, 192],
    [192, 160, 192],
    [0,  96,  64],
    [128,  96,  64],
    [0, 224,  64],
    [128, 224,  64],
    [0,  96, 192],
    [128,  96, 192],
    [0, 224, 192],
    [128, 224, 192],
    [64,  96,  64],
    [192,  96,  64],
    [64, 224,  64],
    [192, 224,  64],
    [64,  96, 192],
    [192,  96, 192],
    [64, 224, 192],
    [192, 224, 192],
    [32,  32,   0],
    [160,  32,   0],
    [32, 160,   0],
    [160, 160,   0],
    [32,  32, 128],
    [160,  32, 128],
    [32, 160, 128],
    [160, 160, 128],
    [96,  32,   0],
    [224,  32,   0],
    [96, 160,   0],
    [224, 160,   0],
    [96,  32, 128],
    [224,  32, 128],
    [96, 160, 128],
    [224, 160, 128],
    [32,  96,   0],
    [160,  96,   0],
    [32, 224,   0],
    [160, 224,   0],
    [32,  96, 128],
    [160,  96, 128],
    [32, 224, 128],
    [160, 224, 128],
    [96,  96,   0],
    [224,  96,   0],
    [96, 224,   0],
    [224, 224,   0],
    [96,  96, 128],
    [224,  96, 128],
    [96, 224, 128],
    [224, 224, 128],
    [32,  32,  64],
    [160,  32,  64],
    [32, 160,  64],
    [160, 160,  64],
    [32,  32, 192],
    [160,  32, 192],
    [32, 160, 192],
    [160, 160, 192],
    [96,  32,  64],
    [224,  32,  64],
    [96, 160,  64],
    [224, 160,  64],
    [96,  32, 192],
    [224,  32, 192],
    [96, 160, 192],
    [224, 160, 192],
    [32,  96,  64],
    [160,  96,  64],
    [32, 224,  64],
    [160, 224,  64],
    [32,  96, 192],
    [160,  96, 192],
    [32, 224, 192],
    [160, 224, 192],
    [96,  96,  64],
    [224,  96,  64],
    [96, 224,  64],
    [224, 224,  64],
    [96,  96, 192],
    [224,  96, 192],
    [96, 224, 192],
    [224, 224, 192]
]) / 256.


class LinearGaussianWithBirthEstimation:
    def __init__(self, est):
        self.gm = est.gm
        if hasattr(est, 'label'):
            self.label = est.label

    def visualize(self, ax, color, label):
        if hasattr(self, 'label'):
            for i in range(len(self.label)):
                ax.scatter(self.gm.m[i, 0], self.gm.m[i, 2],
                           c=[COLOR[self.label[i]]], s=10)
                draw_ellipse(self.gm.m[i][[0, 2]], self.gm.P[i][[0, 2]][:, [0, 2]],
                             ax, color=COLOR[self.label[i]], fill=None, linestyle='solid')
                ax.arrow(*self.gm.m[i][[0, 2]], *3 *
                         self.gm.m[i][[1, 3]],
                         color=COLOR[self.label[i]])

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
