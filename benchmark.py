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
                    choices=['GM-Bernoulli', 'GM-PHD',
                             'GM-CPHD', 'GM-GLMB', 'GM-LMB', 'SMC-PHD'],
                    required=True,
                    help='filter to be used')
parser.add_argument('-n', '--nruns',
                    type=int, default=100,
                    help='number of runs (default: 100)')
parser.add_argument('-w', '--warmups',
                    type=int, default=10,
                    help='number of runs for warmups (default: 10)')
args = parser.parse_args()

track_single = args.single
model_id = args.model
filter_id = args.filter
nruns = args.nruns
warmups = args.warmups

# ======================================

model = gen_model(track_single, model_id)
filt = gen_filter(filter_id, model)

# ======================================

print('Generating examples...')
scenarios = []
for _ in tqdm(range(nruns + 1)):
    truth = model.gen_truth()
    obs = model.gen_obs(truth)
    scenarios.append(obs)
print('Generation done!')
print('================')

# ======================================

print('Warming up...')
for _ in tqdm(range(warmups)):
    filt.run(scenarios[0].Z)
print('Warming up done!')

# =============================

print('Benchmarking...')
meter = []
bar = tqdm(scenarios[1:])
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

# ======================================

print('Average time:', np.mean(meter))
print('Std time:', np.std(meter))
