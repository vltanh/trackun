from gen_model import gen_model
from gen_truth import gen_truth
from gen_meas import gen_meas
from run_filter import run_filter

import time

from tqdm import tqdm
import numpy as np

np.random.seed(3698)

nruns = 100
meter = []

bar = tqdm(range(nruns))
for i in bar:
    model = gen_model()
    truth = gen_truth(model)
    meas = gen_meas(model, truth)

    now = time.time()
    run_filter(model, meas)
    elapsed = time.time() - now

    meter.append(elapsed)
    bar.set_description_str(
        f'Avg time: {np.mean(meter):.4f} ' +
        f'(+/- {np.std(meter):.4f})'
    )

print('=================')
print('Average time:', np.mean(meter))
print('Std time:', np.std(meter))
