import logging

from .ntu_reader import NTU_Reader
from .ntu_transform import NTU_Seq_Transform

__generator = {
    'ntu60': NTU_Reader,
    'ntu60-transform': NTU_Seq_Transform
}


def create(args):
    dataset = args.dataset
    dataset_args = args.dataset_args
    if dataset not in __generator.keys():
        logging.info('')
        logging.error('Error: Do NOT exist this dataset: {}!'.format(dataset))
        raise ValueError()
    return __generator[dataset](**dataset_args)

    # The flags to decide what denoising we want, should be included in the dataset_args variable.
