from examples.models import *
from trackun.filters import *

from time import time

from tqdm import tqdm
import numpy as np
np.random.seed(3698)

card_mode = ['single', 'multi'][0]
model_mode = ['linear_gaussian'][0]
filter_name = ['GM-Bernoulli', 'GM-PHD', 'GM-CPHD'][0]
nruns = 100

if card_mode == 'multi':
    if model_mode == 'linear_gaussian':
        model = LinearGaussianWithBirthModel()
elif card_mode == 'single':
    if model_mode == 'linear_gaussian':
        model = SingleObjectLinearGaussianWithBirthModel()
scenarios = []
for _ in range(nruns):
    truth = model.gen_truth()
    obs = model.gen_obs(truth)
    scenarios.append(obs)

if filter_name == 'GM-Bernoulli':
    filt = Bernoulli_GMS_Filter(model)
elif filter_name == 'GM-PHD':
    filt = PHD_GMS_Filter(model)
elif filter_name == 'GM-CPHD':
    filt = CPHD_GMS_Filter(model)

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
