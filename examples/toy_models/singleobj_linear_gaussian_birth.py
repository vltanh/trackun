import numpy as np

from trackun.models.motion import ConstantVelocityGaussianMotionModel
from trackun.models.measurement import ConstantVelocityGaussianMeasurementModel
from trackun.models.clutter import UniformClutterModel
from trackun.models.birth import MultiBernoulliMixtureGaussianBirthModel
from trackun.models.survival import ConstantSurvivalModel
from trackun.models.detection import ConstantDetectionModel

from examples.generate_data import Observation, Truth

__all__ = [
    'SingleObjectLinearGaussianWithBirthModel',
]


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

        # State/Observation dimensions
        self.x_dim = self.motion_model.x_dim
        self.z_dim = self.measurement_model.z_dim

        # Birth model
        self.birth_model = MultiBernoulliMixtureGaussianBirthModel()
        self.birth_model.add(r=0.03,
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
        ])
        tbirth = np.array([
            10,
        ])
        tdeath = np.array([
            80,
        ])

        truth = Truth(100)
        truth.generate(self, xstart, tbirth, tdeath)
        return truth

    def gen_obs(self, truth):
        return Observation(self, truth)
