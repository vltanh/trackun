from examples.models import *
from src.filters import *

from time import time

from tqdm import tqdm
import numpy as np
np.random.seed(3698)

nruns = 100

model = LinearGaussianWithBirthModel()
scenarios = []
for _ in range(nruns):
    truth = model.gen_truth()
    obs = model.gen_obs(truth)
    scenarios.append(obs)

filt = PHD_GMS_Filter(model)

meter = []
bar = tqdm(scenarios)
for obs in bar:
    Z = obs.Z

    now = time()
    filt.run(Z)
    elapsed = time() - now

    meter.append(elapsed)

    bar.set_description_str(
        f'Avg time: {np.mean(meter):.4f} ' +
        f'(+/- {np.std(meter):.4f})'
    )

print('=================')
print('Average time:', np.mean(meter))
print('Std time:', np.std(meter))
