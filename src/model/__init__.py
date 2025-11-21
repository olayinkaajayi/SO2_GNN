from . import TransGCN
from . import MPGCN

__models = {
    'Trans-GCN': TransGCN,
    'MPGCN': MPGCN
}

def create(model_name, **kwargs):
    return __models[model_name].create(**kwargs)
