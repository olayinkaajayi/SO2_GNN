import numpy as np
import torch
from torch.utils.data import Dataset
from . import tools
import logging


class NTU_Feeder(Dataset):
    def __init__(self, phase, graph, data_path, p_interval=1, p_interval_test=1, random_shift=False,
                 random_move=False, random_rot=None, window_size=-1, normalization=False,
                 vel=False, sort=False, load_part=False, load_portion=1.0, **kwargs):
        """
        :param data_path:
        :param phase: training set or test set
        :param random_shift: If true, randomly pad zeros at the begining or end of sequence
        :param random_move:
        :param random_rot: rotate skeleton around xyz axis
        :param window_size: The length of the output sequence
        :param normalization: If true, normalize input sequence
        :param vel: use motion modality or not
        """
        self.graph = graph
        self.data_path = data_path
        self.phase = phase
        self.random_shift = random_shift
        self.random_move = random_move
        self.window_size = window_size
        self.p_interval = p_interval if phase == "train" else p_interval_test
        self.random_rot = random_rot[0] if (random_rot is not None) and (phase == "train") else False
        self.rand_rot_portion = random_rot[1]
        self.vel = vel
        self.load_part = load_part
        self.load_portion = load_portion
        self.load_data()

        #### Rotation angle:
        self.angle = 0.3 # in radians

        if self.random_rot:
            self.rotate_portion()
            logging.info(f'\nRotating only {self.rand_rot_portion:.0%} of training data...')


        if sort:
            self.get_n_per_class()
            self.sort()
        if normalization:
            self.get_mean_map()

    def rotate_portion(self):
        """This script creates a list of indices to be rotated"""
        N = len(self.data)
        if self.rand_rot_portion != 1:
            self.rotate_list = set((np.random.choice(a=range(N), size=int(N*self.rand_rot_portion), replace=False)))
        else:
            self.rotate_list = set(np.arange(N))

    def load_partial_data(self, strat=False):
         
         if strat:
             self.load_partial_data_strat()
             return

         N = len(self.data)
         sub_list = np.random.choice(a=range(N), size=int(N*self.load_portion), replace=False)
         tmp = self.data
         self.data = tmp[sub_list]
         tmp = self.label
         self.label = tmp[sub_list]

    def load_partial_data_strat(self):
        """This stratefied subsampling ensures we have all classes present in our partial dataset."""
        unique_classes = np.unique(self.label)
        selected_indices = []

        for c in unique_classes:
            class_indices = np.where(self.label == c)[0]
            n_class = len(class_indices)

            # number to sample from this class
            n_sample = max(1, int(n_class * self.load_portion))

            class_sub = np.random.choice(class_indices, size=n_sample, replace=False)
            selected_indices.extend(class_sub)

        selected_indices = np.array(selected_indices)

        self.data = self.data[selected_indices]
        self.label = self.label[selected_indices]

    def load_data(self):
        # data: N C V T M
        npz_data = np.load(self.data_path)
        if self.phase == 'train':
            self.data = npz_data['x_train']
            self.label = np.argmax(npz_data['y_train'], axis=-1)
        elif self.phase == 'eval':
            self.data = npz_data['x_test']
            self.label = np.argmax(npz_data['y_test'], axis=-1)
        else:
            raise NotImplementedError('data phase only supports train/eval')
        nan_out = np.isnan(self.data.mean(-1).mean(-1))==False
        self.data = self.data[nan_out]
        self.label = self.label[nan_out]
        self.sample_name = [self.phase + '_' + str(i) for i in range(len(self.data))]

        if self.load_part: # loads only a portion of the data
            self.load_partial_data()
            logging.warning(f'Using {self.load_portion:.2%} of the {self.phase.upper()} dataset')

        N, T, _ = self.data.shape
        self.data = self.data.reshape((N, T, 2, 25, 3)).transpose(0, 4, 1, 3, 2) # shape: N, C, T, V, M

        # num_input, num_channel, _, _, _ : dimension info used by model
        N,C,T,V,M = self.data.shape
        self.datashape = [1, C, self.window_size, V, M] # 1 corresponds to one modality: joint
        

    def get_n_per_class(self):
        self.n_per_cls = np.zeros(len(self.label), dtype=int)
        for label in self.label:
            self.n_per_cls[label] += 1
        self.csum_n_per_cls = np.insert(np.cumsum(self.n_per_cls), 0, 0)

    def sort(self):
        sorted_idx = self.label.argsort()
        self.data = self.data[sorted_idx]
        self.label = self.label[sorted_idx]

    def get_mean_map(self):
        data = self.data
        N, C, T, V, M = data.shape
        self.mean_map = data.mean(axis=2, keepdims=True).mean(axis=4, keepdims=True).mean(axis=0)
        self.std_map = data.transpose((0, 2, 4, 1, 3)).reshape((N * T * M, C * V)).std(axis=0).reshape((C, 1, V, 1))

    def __len__(self):
        return len(self.label)

    def __iter__(self):
        return self

    def __getitem__(self, index):
        # Ensure that data_numpy: Tx(MVC) is reshaped to CxTxVxM
        data_numpy = self.data[index]
        label = self.label[index]

        data_numpy = np.array(data_numpy) # C, T, V, M
        valid_frame_num = np.sum(data_numpy.sum(0).sum(-1).sum(-1) != 0)
        
        data_numpy = tools.valid_crop_resize(data_numpy, valid_frame_num, self.p_interval, self.window_size) # shape: C,T,V,M
        if self.random_rot and (index in self.rotate_list):
            data_numpy = tools.random_rot(data_numpy, theta=self.angle) # shape: C,T,V,M
        if self.vel:
            data_numpy[:, :-1] = data_numpy[:, 1:] - data_numpy[:, :-1]
            data_numpy[:, -1] = 0

        if isinstance(data_numpy, np.ndarray):
            data_numpy = torch.from_numpy(data_numpy)

        # for the model, input needs to be I,C,T,V,M (where I is number of inputs: joint, bone, motion etc)
        # we set I = 1 i.e. only joint as input
        data_numpy = data_numpy.unsqueeze(0) # shape: I,C,T,V,M

        return data_numpy, label

    def top_k(self, score, top_k):
        rank = score.argsort()
        hit_top_k = [l in rank[i, -top_k:] for i, l in enumerate(self.label)]
        return sum(hit_top_k) * 1.0 / len(hit_top_k)


def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod