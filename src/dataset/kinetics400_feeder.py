import logging

import numpy as np
import torch
from torch.utils.data import Dataset


class Kinetics400Feeder(Dataset):
    """Load ST-GCN-compatible OpenPose-18 Kinetics-400 sequences from an NPZ archive."""

    num_classes = 400

    def __init__(self, phase, graph, data_path, window_size=300, p_interval=(1.0,),
                 p_interval_test=(1.0,), vel=False, **kwargs):
        if phase not in ('train', 'eval'):
            raise NotImplementedError('data phase only supports train/eval')
        if graph.num_node != 18:
            raise ValueError(
                f'Kinetics-400 ST-GCN poses require an 18-node graph, got {graph.num_node}'
            )
        if window_size <= 0:
            raise ValueError('window_size must be a positive integer')

        self.phase = phase
        self.window_size = window_size
        self.p_interval = p_interval if phase == 'train' else p_interval_test
        self.vel = vel

        with np.load(data_path, allow_pickle=False) as archive:
            data_key = f'x_{phase}'
            label_key = f'y_{phase}'
            missing_keys = [
                key for key in (data_key, label_key, 'class_names') if key not in archive
            ]
            if missing_keys:
                raise ValueError(
                    f'{data_path} is missing required archive key(s): {", ".join(missing_keys)}'
                )
            self.data = np.asarray(archive[data_key], dtype=np.float32)
            self.label = np.asarray(archive[label_key])
            class_names = np.asarray(archive['class_names'])

        if (class_names.shape != (self.num_classes,) or
                class_names.dtype.kind not in ('U', 'S') or
                len(np.unique(class_names)) != self.num_classes or
                any(not name.strip() for name in class_names.tolist())):
            raise ValueError('class_names must contain 400 unique, non-empty class names')

        if (self.data.ndim != 5 or self.data.shape[1] != 3 or
                self.data.shape[3:] != (18, 2)):
            raise ValueError(
                f'{data_key} must have shape [N, 3, T, 18, 2], got {self.data.shape}'
            )
        if self.data.shape[2] != self.window_size:
            raise ValueError(
                f'{data_key} has {self.data.shape[2]} source frames; '
                f'window_size must match ST-GCN input ({self.window_size})'
            )
        if self.label.ndim != 1 or len(self.label) != len(self.data):
            raise ValueError(f'{label_key} must be a 1D label array matching {data_key}')
        if not np.issubdtype(self.label.dtype, np.integer):
            raise ValueError(f'{label_key} must contain integer class IDs in [0, 399]')
        if np.any(self.label < 0) or np.any(self.label >= self.num_classes):
            raise ValueError(f'{label_key} contains class IDs outside [0, 399]')
        if not np.isfinite(self.data).all():
            raise ValueError(f'{data_key} contains NaN or infinite coordinates')
        if len(self.data) == 0:
            raise ValueError(f'{data_key} must contain at least one sample')

        self.datashape = [1, 3, self.window_size, 18, 2]
        logging.info('Loaded %d %s Kinetics-400 pose sequences', len(self.data), phase)

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        data_numpy = self.data[index].copy()
        if self.vel:
            data_numpy[:, :-1] = data_numpy[:, 1:] - data_numpy[:, :-1]
            data_numpy[:, -1] = 0

        return torch.from_numpy(data_numpy).unsqueeze(0), int(self.label[index])