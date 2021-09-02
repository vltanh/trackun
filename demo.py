from examples.models import *
from examples.visualize import visualize
from trackun.filters import *

from time import time

import numpy as np
np.random.seed(3698)

filters_name = ['GM-CPHD', 'GM-PHD']

print('Begin generating examples...')
model = LinearGaussianWithBirthModel()
truth = model.gen_truth()
obs = model.gen_obs(truth)
print('Generation done!')
print('================')

print('Begin filtering...')
filters = [
    CPHD_GMS_Filter(model),
    PHD_GMS_Filter(model)
]

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
