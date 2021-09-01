from examples.models import *
from examples.visualize import visualize
from src.filters import *

import numpy as np
np.random.seed(3698)

model = LinearGaussianWithBirthModel()
truth = model.gen_truth()
obs = model.gen_obs(truth)

filt = PHD_GMS_Filter(model)
w_upds, m_upds, P_upds = filt.run(obs.Z)

visualize(w_upds, m_upds, P_upds, model, obs, truth)
