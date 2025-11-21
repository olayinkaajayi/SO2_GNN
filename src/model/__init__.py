from . import TransGCN

__models = {
    'Trans-GCN': TransGCN
}

def create(model_name, **kwargs):
    return __models[model_name].create(**kwargs)
