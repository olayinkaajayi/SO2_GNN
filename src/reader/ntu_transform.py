# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
import sys
import os
import os.path as osp
import numpy as np
import pickle
import logging

from sklearn.model_selection import train_test_split
from utils import create_aligned_dataset

###--------------------------------------------------------------------------------
# From the InfoGCN pre-processing script, this is the script to "transform" the sequence.
# This involves rotation (such that all skeletons face same direction),
# and other such "transformations".
# It also includes a function to make all sample length (time) the same (max_time=300) --> align_skeletons
# There is also a function that does "translation" normalisation, with reference to the first skeleton in the sequence.
# [It is within the Feeder of the dataset that the sample length is trimmed down or upsampled.]
#
# We are using the InfoGCN pre-processing, and would gradually scale back any of the pre-processing I do not want.
#
# My current research work is focused on designing a deep learning model for the skeletons
# without having to perform any/some of those transformations, especially rotation.
###--------------------------------------------------------------------------------


class NTU_Seq_Transform():
    def __init__(self, evaluations, transform_flag, stat_path, data_path, save_path):
        
        stat_path = osp.join(stat_path, 'statistics')
        self.setup_file = osp.join(stat_path, 'setup.txt')
        self.camera_file = osp.join(stat_path, 'camera.txt')
        self.performer_file = osp.join(stat_path, 'performer.txt')
        self.replication_file = osp.join(stat_path, 'replication.txt')
        self.label_file = osp.join(stat_path, 'label.txt')
        self.skes_name_file = osp.join(stat_path, 'skes_available_name.txt')

        denoised_path = osp.join(data_path, 'denoised_data')
        self.raw_skes_joints_pkl = osp.join(denoised_path, 'raw_denoised_joints.pkl')
        self.frames_file = osp.join(denoised_path, 'frames_cnt.txt')

        self.save_path = save_path
        if not osp.exists(self.save_path):
            os.mkdir(self.save_path)

        self.evaluations = evaluations #['CS', 'CV']

        self.seq_transl_flag = transform_flag['translation']
        self.seq_align_flag = transform_flag['orient']


    def remove_nan_frames(self, ske_name, ske_joints, nan_logger):
        num_frames = ske_joints.shape[0]
        valid_frames = []

        for f in range(num_frames):
            if not np.any(np.isnan(ske_joints[f])):
                valid_frames.append(f)
            else:
                nan_indices = np.where(np.isnan(ske_joints[f]))[0]
                nan_logger.info('{}\t{:^5}\t{}'.format(ske_name, f + 1, nan_indices))

        return ske_joints[valid_frames]

    def seq_translation(self, skes_joints):
        """This function performs normalisation."""

        if not self.seq_transl_flag:
            return skes_joints

        for idx, ske_joints in enumerate(skes_joints):
            num_frames = ske_joints.shape[0]
            num_bodies = 1 if ske_joints.shape[1] == 75 else 2
            if num_bodies == 2:
                missing_frames_1 = np.where(ske_joints[:, :75].sum(axis=1) == 0)[0]
                missing_frames_2 = np.where(ske_joints[:, 75:].sum(axis=1) == 0)[0]
                cnt1 = len(missing_frames_1)
                cnt2 = len(missing_frames_2)

            i = 0  # get the "real" first frame of actor1
            while i < num_frames:
                if np.any(ske_joints[i, :75] != 0):
                    break
                i += 1

            origin = np.copy(ske_joints[i, 3:6])  # new origin: joint-2

            for f in range(num_frames):
                if num_bodies == 1:
                    ske_joints[f] -= np.tile(origin, 25)
                else:  # for 2 actors
                    ske_joints[f] -= np.tile(origin, 50)

            if (num_bodies == 2) and (cnt1 > 0):
                ske_joints[missing_frames_1, :75] = np.zeros((cnt1, 75), dtype=np.float32)

            if (num_bodies == 2) and (cnt2 > 0):
                ske_joints[missing_frames_2, 75:] = np.zeros((cnt2, 75), dtype=np.float32)

            skes_joints[idx] = ske_joints  # Update

        return skes_joints


    def frame_translation(self, skes_joints, skes_name, frames_cnt):
        nan_logger = logging.getLogger('nan_skes')
        nan_logger.setLevel(logging.INFO)
        nan_logger.addHandler(logging.FileHandler("./nan_frames.log"))
        nan_logger.info('{}\t{}\t{}'.format('Skeleton', 'Frame', 'Joints'))

        for idx, ske_joints in enumerate(skes_joints):
            num_frames = ske_joints.shape[0]
            # Calculate the distance between spine base (joint-1) and spine (joint-21)
            j1 = ske_joints[:, 0:3]
            j21 = ske_joints[:, 60:63]
            dist = np.sqrt(((j1 - j21) ** 2).sum(axis=1))

            for f in range(num_frames):
                origin = ske_joints[f, 3:6]  # new origin: middle of the spine (joint-2)
                if (ske_joints[f, 75:] == 0).all():
                    ske_joints[f, :75] = (ske_joints[f, :75] - np.tile(origin, 25)) / \
                                        dist[f] + np.tile(origin, 25)
                else:
                    ske_joints[f] = (ske_joints[f] - np.tile(origin, 50)) / \
                                    dist[f] + np.tile(origin, 50)

            ske_name = skes_name[idx]
            ske_joints = self.remove_nan_frames(ske_name, ske_joints, nan_logger)
            frames_cnt[idx] = num_frames  # update valid number of frames
            skes_joints[idx] = ske_joints

        return skes_joints, frames_cnt


    def align_frames(self, skes_joints, frames_cnt):
        """
        Align all sequences with the same frame length.

        """
        num_skes = len(skes_joints)
        max_num_frames = frames_cnt.max()  # 300
        aligned_skes_joints = np.zeros((num_skes, max_num_frames, 150), dtype=np.float32)

        for idx, ske_joints in enumerate(skes_joints):
            num_frames = ske_joints.shape[0]
            num_bodies = 1 if ske_joints.shape[1] == 75 else 2

            if num_bodies == 1:
                # np.zeros_like(ske_joints) gives zeros to the second actor
                aligned_skes_joints[idx, :num_frames] = np.hstack((ske_joints,
                                                                np.zeros_like(ske_joints)))
            else:
                aligned_skes_joints[idx, :num_frames] = ske_joints

        return aligned_skes_joints # size: N x max_num_frames x 150


    def one_hot_vector(self, labels):
        num_skes = len(labels)
        labels_vector = np.zeros((num_skes, 60))
        for idx, l in enumerate(labels):
            labels_vector[idx, l] = 1

        return labels_vector


    def split_train_val(self, train_indices, method='sklearn', ratio=0.05):
        """
        Get validation set by splitting data randomly from training set with two methods.
        In fact, I thought these two methods are equal as they got the same performance.

        """
        if method == 'sklearn':
            return train_test_split(train_indices, test_size=ratio, random_state=10000)
        else:
            np.random.seed(10000)
            np.random.shuffle(train_indices)
            val_num_skes = int(np.ceil(0.05 * len(train_indices)))
            val_indices = train_indices[:val_num_skes]
            train_indices = train_indices[val_num_skes:]
            return train_indices, val_indices


    def split_dataset(self, skes_joints, label, performer, camera, evaluation, save_path):
        train_indices, test_indices = self.get_indices(performer, camera, evaluation)
        m = 'sklearn'  # 'sklearn' or 'numpy'
        # Select validation set from training set
        # train_indices, val_indices = split_train_val(train_indices, m)

        # Save labels and num_frames for each sequence of each data set
        train_labels = label[train_indices]
        test_labels = label[test_indices]

        train_x = skes_joints[train_indices]
        train_y = self.one_hot_vector(train_labels)
        test_x = skes_joints[test_indices]
        test_y = self.one_hot_vector(test_labels)

        save_name = osp.join(save_path,'NTU60_%s.npz' % evaluation)
        np.savez(save_name, x_train=train_x, y_train=train_y, x_test=test_x, y_test=test_y)


    def get_indices(self, performer, camera, evaluation='CS'):
        test_indices = np.empty(0)
        train_indices = np.empty(0)

        if evaluation == 'CS':  # Cross Subject (Subject IDs)
            train_ids = [1,  2,  4,  5,  8,  9,  13, 14, 15, 16,
                        17, 18, 19, 25, 27, 28, 31, 34, 35, 38]
            test_ids = [3,  6,  7,  10, 11, 12, 20, 21, 22, 23,
                        24, 26, 29, 30, 32, 33, 36, 37, 39, 40]

            # Get indices of test data
            for idx in test_ids:
                temp = np.where(performer == idx)[0]  # 0-based index
                test_indices = np.hstack((test_indices, temp)).astype(int)

            # Get indices of training data
            for train_id in train_ids:
                temp = np.where(performer == train_id)[0]  # 0-based index
                train_indices = np.hstack((train_indices, temp)).astype(int)
        else:  # Cross View (Camera IDs)
            train_ids = [2, 3]
            test_ids = 1
            # Get indices of test data
            temp = np.where(camera == test_ids)[0]  # 0-based index
            test_indices = np.hstack((test_indices, temp)).astype(int)

            # Get indices of training data
            for train_id in train_ids:
                temp = np.where(camera == train_id)[0]  # 0-based index
                train_indices = np.hstack((train_indices, temp)).astype(int)

        return train_indices, test_indices



    def gendata(self):
        """
            We call the function that applies the specific
            skeleton trasnformation (normalisation) techniques we want.
        """
        camera = np.loadtxt(self.camera_file, dtype=int)  # camera id: 1, 2, 3
        performer = np.loadtxt(self.performer_file, dtype=int)  # subject id: 1~40
        label = np.loadtxt(self.label_file, dtype=int) - 1  # action label: 0~59

        frames_cnt = np.loadtxt(self.frames_file, dtype=int)  # frames_cnt

        with open(self.raw_skes_joints_pkl, 'rb') as fr:
            skes_joints = pickle.load(fr)  # a list

        skes_joints = self.seq_translation(skes_joints)

        skes_joints = self.align_frames(skes_joints, frames_cnt)  # aligned to the same frame length
                                                                  # size: N x max_num_frames x 150

        for evaluation in self.evaluations:
            self.split_dataset(skes_joints, label, performer, camera, evaluation, self.save_path)

            file = osp.join(self.save_path,'NTU60_%s.npz' % evaluation)
            create_aligned_dataset(file_list=[file], align=self.seq_align_flag)


        
    def start(self):    
        logging.info(f'Phase: Train and Test data at once.')
        self.gendata()
                
