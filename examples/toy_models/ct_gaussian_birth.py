from trackun.models.motion import CoordinatedTurnGaussianMotionModel
from trackun.models.measurement import BearingGaussianMeasurementModel
from trackun.models.clutter import UniformClutterModel
from trackun.models.birth import MultiBernoulliGaussianBirthModel
from trackun.models.survival import ConstantSurvivalModel
from trackun.models.detection import BearingGaussianDetectionModel
from examples.generate_data import Observation, Truth

import numpy as np

__all__ = [
    'CTGaussianWithBirthModel',
]


class CTGaussianWithBirthModel:
    def __init__(self) -> None:
        # Basic parameters

        # Motion model
        Q = np.eye(3)
        self.motion_model = \
            CoordinatedTurnGaussianMotionModel(5, Q, 5., np.pi/180)

        # Measurement model
        R = np.diag([2 * np.pi / 180., 10.]) ** 2
        self.measurement_model = BearingGaussianMeasurementModel(2, R)

        # Dimensions
        self.x_dim = self.motion_model.x_dim  # state dimension
        self.v_dim = self.motion_model.v_dim  # process noise dimension
        self.z_dim = self.measurement_model.z_dim  # observation dimension
        self.w_dim = self.measurement_model.w_dim  # observation noise dimension

        # Birth model
        self.birth_model = MultiBernoulliGaussianBirthModel()
        P0 = np.diag([50, 50, 50, 50, 6 * np.pi / 180.]) ** 2
        self.birth_model.add(w=.03, m=[-1500., 0., 250., 0., 0.], P=P0)
        self.birth_model.add(w=.03, m=[-250., 0., 1000., 0., 0.], P=P0)
        self.birth_model.add(w=.03, m=[250., 0., 750., 0., 0.], P=P0)
        self.birth_model.add(w=.03, m=[1000., 0., 1500., 0., 0.], P=P0)

        # Survival model
        self.survival_model = ConstantSurvivalModel(0.99)

        # Detection model
        m = np.zeros(2)
        P = np.diag([2000., 2000.]) ** 2
        self.detection_model = \
            BearingGaussianDetectionModel(0.98, m, P)

        # Clutter model
        range_c = np.array([
            [-np.pi / 2., np.pi / 2.],
            [0, 2000]
        ])
        self.clutter_model = UniformClutterModel(10, range_c)

    def gen_truth(self):
        w_turn = 2 * np.pi / 180

        xstart = np.array([
            [1000+3.8676, -10, 1500-11.7457, -10, w_turn/8],
            [-250-5.8857,  20, 1000+11.4102, 3, -w_turn/3],
            [-1500-7.3806, 11, 250+6.7993, 10, -w_turn/2],
            [-1500, 43, 250, 0, 0],
            [250-3.8676, 11, 750-11.0747, 5, w_turn/4],
            [-250+7.3806, -12, 1000-6.7993, -12, w_turn/2],
            [1000, 0, 1500, -10, w_turn/4],
            [250, -50, 750, 0, -w_turn/4],
            [1000, -50, 1500, 0, -w_turn/4],
            [250, -40, 750, 25, w_turn/4],
        ])[:, np.newaxis, :]
        tbirth = np.array([
            1,
            10, 10, 10,
            20,
            40, 40, 40,
            60, 60,
        ])
        tdeath = np.array([
            100,
            100, 100, 66,
            80,
            100, 100, 80,
            100, 100,
        ])
        truth = Truth(100)
        truth.generate(self, xstart, tbirth, tdeath)
        return truth

    def gen_obs(self, truth):
        return Observation(self, truth)
