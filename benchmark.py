from trackun.filters import *
from trackun.models import *

from time import time

from tqdm import tqdm
import numpy as np
np.random.seed(3698)

card_mode = ['single', 'multi'][1]
model_mode = ['linear_gaussian', 'ct_gaussian'][0]
filter_name = ['GM-Bernoulli', 'GM-PHD', 'GM-CPHD', 'SMC-PHD'][2]
nruns = 100


def gen_model(card_mode, model_mode):
    if card_mode == 'multi':
        if model_mode == 'linear_gaussian':
            model = LinearGaussianWithBirthModel()
        elif model_mode == 'ct_gaussian':
            model = CTGaussianWithBirthModel()
        else:
            raise Exception('Unknown model mode.')
    elif card_mode == 'single':
        if model_mode == 'linear_gaussian':
            model = SingleObjectLinearGaussianWithBirthModel()
        else:
            raise Exception('Unknown model mode.')
    else:
        raise Exception('Unknown cardinality mode.')
    return model


def gen_filter(name, model):
    if name == 'GM-CPHD':
        filt = CPHD_GMS_Filter(model)
    elif name == 'GM-PHD':
        filt = PHD_GMS_Filter(model)
    elif name == 'GM-Bernoulli':
        filt = Bernoulli_GMS_Filter(model)
    elif name == 'SMC-PHD':
        filt = PHD_SMC_Filter(model)
    else:
        raise Exception('Unknown filter name.')
    return filt


model = gen_model(card_mode, model_mode)
scenarios = []
for _ in range(nruns):
    truth = model.gen_truth()
    obs = model.gen_obs(truth)
    scenarios.append(obs)

filt = gen_filter(filter_name, model)
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
