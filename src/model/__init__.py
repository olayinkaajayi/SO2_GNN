from . import TransGCN
from . import MPGCN

__models = {
    'Trans-GCN': TransGCN,
    'MPGCN': MPGCN,
    'InfoGCN': MPGCN, # define this model
    'ST-Trans': MPGCN # define this model
}

def create(model_name, **kwargs):
    return __models[model_name].create(**kwargs)
