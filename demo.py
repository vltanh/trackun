from examples.utils import gen_model, gen_filter
from examples.visualize import visualize

import pickle
import argparse

from tqdm import tqdm
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
                    choices=[
                        'GM-Bernoulli',
                        'GM-PHD', 'SMC-PHD',
                        'GM-CPHD',
                        'GM-GLMB', 'GM-JointGLMB',
                        'GM-LMB', 'GM-JointLMB',
                        'SORT',
                    ],
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

# ======================================

model = gen_model(track_single, model_id)
filters = [
    gen_filter(filter_id, model)
    for filter_id in filter_ids
]

# ======================================

print('Begin generating examples...')
truth = model.gen_truth()
obs = model.gen_obs(truth)

d = pickle.load(open('tests/linear_multi/truth.pkl', 'rb'))
truth.X = d['X']
truth.track_list = d['track_list']

d = pickle.load(open('tests/linear_multi/meas.pkl', 'rb'))
obs.Z = d['Z']

print('Generation done!')
print('================')

# ======================================

print('Begin filtering...')
Zs = obs.Z
ests = dict()
for n, f in zip(filter_ids, filters):
    upds_k = f.init()
    ests[n] = []
    for Z in tqdm(Zs):
        upds_k = f.step(Z, upds_k)
        ests_k = f.visualizable_estimate(upds_k)
        ests[n].append(model.gen_vis_obj(ests_k))
print('Filtering done!')
print('================')

# ======================================

print('Begin visualizing...')
visualize(ests, model, obs, truth, output_dir)
print('Visualization done!')
