from gen_model import gen_model
from gen_truth import gen_truth
from gen_meas import gen_meas
from run_filter import run_filter
from visualize import visualize

import numpy as np
np.random.seed(3698)

model = gen_model()

truth = gen_truth(model)

meas = gen_meas(model, truth)

w_upds, m_upds, P_upds = run_filter(model, meas)

visualize(w_upds, m_upds, P_upds, model, meas, truth)
