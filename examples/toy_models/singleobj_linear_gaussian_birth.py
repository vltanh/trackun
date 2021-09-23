from trackun.models.motion import ConstantVelocityGaussianMotionModel
from trackun.models.measurement import ConstantVelocityGaussianMeasurementModel
from trackun.models.clutter import UniformClutterModel
from trackun.models.birth import MultiBernoulliMixtureGaussianBirthModel
from trackun.models.survival import ConstantSurvivalModel
from trackun.models.detection import ConstantDetectionModel
from trackun.metrics.ospa import OSPA
from examples.generate_data import Observation, Truth
from examples.visualize import plot_2d_gaussian_mixture

import numpy as np

__all__ = [
    'SingleObjectLinearGaussianWithBirthModel',
]


class SingleObjectLinearGaussianWithBirthObservation(Observation):
    def get_obs(self, k):
        Zs = self.Z[k]
        X = Zs[:, 0]
        Y = Zs[:, 1]
        return X, Y


class SingleObjectLinearGaussianWithBirthEstimation:
    def __init__(self, est):
        self.gm = est.gm

    def visualize(self, ax, color, label):
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


class SingleObjectLinearGaussianWithBirthModel:
    def __init__(self) -> None:
        # Motion model
        self.motion_model = \
            ConstantVelocityGaussianMotionModel(dim=2,
                                                noise_std=5)

        # Measurement model
        R = np.diag([100., 100.])
        self.measurement_model = \
            ConstantVelocityGaussianMeasurementModel(dim=2,
                                                     noise_cov=R)

        # Dimensions
        self.x_dim = self.motion_model.x_dim
        self.z_dim = self.measurement_model.z_dim

        # Birth model
        self.birth_model = MultiBernoulliMixtureGaussianBirthModel()
        self.birth_model.add(r=0.3,
                             ws=[1.],
                             ms=[[0., 0., 0., 0.]],
                             Ps=[np.diag([1000., 10., 1000., 10.]) ** 2])

        # Survival model
        self.survival_model = ConstantSurvivalModel(.99)

        # Detection paramters
        self.detection_model = ConstantDetectionModel(.98)

        # Clutter model
        range_c = np.array([
            [-1000, 1000],
            [-1000, 1000]
        ])
        self.clutter_model = UniformClutterModel(60, range_c)

    def gen_truth(self):
        xstart = np.array([
            [0., 3., 0., 9.],
            [200., 1., 500., 3.],
        ])
        tbirth = np.array([
            10,
            50,
        ])
        tdeath = np.array([
            40,
            95,
        ])

        truth = Truth(100)
        truth.generate(self, xstart, tbirth, tdeath)
        return truth

    def gen_obs(self, truth):
        return SingleObjectLinearGaussianWithBirthObservation(self, truth)

    def gen_vis_obj(self, est):
        return SingleObjectLinearGaussianWithBirthEstimation(est)

    def get_vis_lim(self):
        return (-1000, 1000), (-1000, 1000)
