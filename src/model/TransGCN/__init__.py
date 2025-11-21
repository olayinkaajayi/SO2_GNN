from .nets import TransGCN

def create(**kwargs):
    return TransGCN(**kwargs)
