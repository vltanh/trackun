from examples.visualize import visualize
from trackun.models import *
from trackun.filters import *

from time import time

import numpy as np
np.random.seed(3698)

card_mode = 'multi'
model_mode = 'linear_gaussian'
filters_name = ['GM-CPHD', 'GM-PHD']


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


print('Begin generating examples...')
model = gen_model(card_mode, model_mode)
truth = model.gen_truth()
obs = model.gen_obs(truth)
print('Generation done!')
print('================')

print('Begin filtering...')
filters = [gen_filter(name, model) for name in filters_name]

Z = obs.Z
ests = []
for n, f in zip(filters_name, filters):
    start = time()
    est = f.run(Z)
    elapsed = time() - start

    ests.append(est)
    print(f'[{n}] Done! Took {elapsed} (s)')
print('Filtering done!')
print('================')

print('Begin visualizing...')
visualize(
    *list(zip(*ests)),
    filters_name,
    model, obs, truth
)
print('Visualization done!')
