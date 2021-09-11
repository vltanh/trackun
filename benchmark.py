from examples.utils import gen_model, gen_filter

from time import time
import argparse

from tqdm import tqdm
import numpy as np
np.random.seed(3698)

parser = argparse.ArgumentParser(description='Benchmark')
parser.add_argument('-s', '--single',
                    action='store_true',
                    help='toggle single object mode')
parser.add_argument('-m', '--model',
                    choices=['linear_gaussian', 'ct_gaussian'],
                    required=True,
                    help='motion/measurement model to be used')
parser.add_argument('-f', '--filter',
                    choices=['GM-Bernoulli', 'GM-PHD', 'GM-CPHD', 'SMC-PHD'],
                    required=True,
                    help='filter to be used')
parser.add_argument('-n', '--nruns',
                    type=int, default=100,
                    help='number of runs (default: 100)')
args = parser.parse_args()

track_single = args.single
model_id = args.model
filter_id = args.filter
nruns = args.nruns

print('Begin generating examples...')
model = gen_model(track_single, model_id)
scenarios = []
for _ in range(nruns):
    truth = model.gen_truth()
    obs = model.gen_obs(truth)
    scenarios.append(obs)
print('Generation done!')
print('================')

print('Benchmarking...')
filt = gen_filter(filter_id, model)
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
print('Benchmarking done!')
print('=================')

print('Result:')
print('\tAverage time:', np.mean(meter))
print('\tStd time:', np.std(meter))
