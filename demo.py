from examples.utils import gen_model, gen_filter
from examples.visualize import visualize

from time import time
import argparse

import numpy as np
np.random.seed(3698)

parser = argparse.ArgumentParser(description='Demonstration')
parser.add_argument('-s', '--single',
                    action='store_true',
                    help='toggle single object mode')
parser.add_argument('-m', '--model',
                    choices=['linear_gaussian', 'ct_gaussian'],
                    required=True,
                    help='motion/measurement model')
parser.add_argument('-f', '--filters',
                    nargs='+',
                    choices=['GM-Bernoulli', 'GM-PHD', 'GM-CPHD', 'SMC-PHD'],
                    required=True,
                    help='filter names')
parser.add_argument('-o', '--output',
                    required=True,
                    help='output directory for visualization')
args = parser.parse_args()

track_single = args.single
model_id = args.model
filter_ids = args.filters
output_dir = args.output


print('Begin generating examples...')
model = gen_model(track_single, model_id)
truth = model.gen_truth()
obs = model.gen_obs(truth)
print('Generation done!')
print('================')

print('Begin filtering...')
filters = [gen_filter(filter_id, model) for filter_id in filter_ids]

Z = obs.Z
ests = []
for n, f in zip(filter_ids, filters):
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
    filter_ids,
    model, obs, truth,
    output_dir
)
print('Visualization done!')
