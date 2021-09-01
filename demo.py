from examples.models import *
from examples.visualize import visualize
from trackun.filters import *

import numpy as np
np.random.seed(3698)

model = LinearGaussianWithBirthModel()
truth = model.gen_truth()
obs = model.gen_obs(truth)

filt1 = CPHD_GMS_Filter(model)
w_upds1, m_upds1, P_upds1 = filt1.run(obs.Z)

filt2 = PHD_GMS_Filter(model)
w_upds2, m_upds2, P_upds2 = filt2.run(obs.Z)

visualize(
    [w_upds1, w_upds2],
    [m_upds1, m_upds2],
    [P_upds1, P_upds2],
    ['GM-CPHD', 'GM-PHD'],
    model, obs, truth
)
