from trackun.models.motion import ConstantVelocityGaussianMotionModel
from trackun.models.measurement import ConstantVelocityGaussianMeasurementModel
from trackun.models.clutter import UniformClutterModel
from trackun.models.birth import MultiBernoulliGaussianBirthModel
from trackun.models.survival import ConstantSurvivalModel
from trackun.models.detection import ConstantDetectionModel
from examples.generate_data import Observation, Truth

import numpy as np

__all__ = [
    'LinearGaussianWithBirthModel',
]


class LinearGaussianWithBirthModel:
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
        self.birth_model = MultiBernoulliGaussianBirthModel()
        P0 = np.diag([10., 10., 10., 10.]) ** 2
        self.birth_model.add(w=.03, m=[0., 0., 0., 0.],       P=P0)
        self.birth_model.add(w=.03, m=[400., 0., -600., 0.],  P=P0)
        self.birth_model.add(w=.03, m=[-800., 0., -200., 0.], P=P0)
        self.birth_model.add(w=.03, m=[-200., 0., 800., 0.],  P=P0)

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
        return Observation(self, truth)
