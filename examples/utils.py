from examples.toy_models import *
from trackun.filters import *


def gen_model(track_single, model_id):
    if not track_single:
        if model_id == 'linear_gaussian':
            model_id = LinearGaussianWithBirthModel()
        elif model_id == 'ct_gaussian':
            model_id = CTGaussianWithBirthModel()
        else:
            raise Exception('Unknown model mode.')
    else:
        if model_id == 'linear_gaussian':
            model_id = SingleObjectLinearGaussianWithBirthModel()
        else:
            raise Exception('Unknown model mode.')
    return model_id


def gen_filter(filter_id, model):
    if filter_id == 'GM-CPHD':
        filt = CPHD_GMS_Filter(model)
    elif filter_id == 'GM-PHD':
        filt = PHD_GMS_Filter(model)
    elif filter_id == 'GM-Bernoulli':
        filt = Bernoulli_GMS_Filter(model)
    elif filter_id == 'SMC-PHD':
        filt = PHD_SMC_Filter(model)
    elif filter_id == 'GM-GLMB':
        filt = GLMB_GMS_Filter(model)
    elif filter_id == 'GM-JointGLMB':
        filt = JointGLMB_GMS_Filter(model)
    elif filter_id == 'GM-LMB':
        filt = LMB_GMS_Filter(model)
    else:
        raise Exception('Unknown filter name.')
    return filt
