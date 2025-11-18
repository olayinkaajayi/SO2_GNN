import logging

from .graphs import Graph
from .ntu_feeder import NTU_Feeder

__data_args = {
    'ntu': {'class': 60, 'feeder': NTU_Feeder},
    'ntu120': {'class': 120, 'feeder': NTU_Feeder}
}


def create(dataset, **kwargs):
    try:
        data_args = __data_args[dataset]
        num_class = data_args['class']
    except:
        logging.info('')
        logging.error('Error: Do NOT exist this dataset: {}!'.format(dataset))
        raise ValueError()
    
    graph = Graph(dataset, **kwargs)
    del kwargs['graph']
    feeders = {
        'train': data_args['feeder'](phase='train', graph=graph, **kwargs),
        'eval': data_args['feeder'](phase='eval', graph=graph, **kwargs),
    }
    data_shape = feeders['train'].datashape

    return feeders, data_shape, num_class, graph.A, graph.parts
