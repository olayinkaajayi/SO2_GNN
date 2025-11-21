import logging
import numpy as np
import os
from torch.utils.data import Dataset
from .utils import graph_processing, multi_input


class NTU_Feeder(Dataset):
    def __init__(self, phase, graph, root_folder, inputs, debug, repeat, max_frame=80, processing='default', person_id=[0,1], input_dims=3, **kwargs):
        self.phase = phase
        self.inputs = inputs
        self.processing = processing
        self.debug = debug
        self.repeat = repeat # repeat skeletons
        
        self.graph = graph.graph
        self.conn = graph.connect_joint
        self.center = graph.center
        self.num_node = graph.num_node
        self.num_person = graph.num_person
        
        self.input_dims = input_dims # expected to be 3 (x,y,z)
        self.max_frame = max_frame
        self.M = len(person_id)
        self.datashape = self.get_datashape()

        data_path = os.path.join(root_folder,'_data.npy')
        # label_path = os.path.join(root_folder, phase+'_label.pkl')

        self.load_data(data_path)


    def load_data(self, data_path):
        # data: N C V T M
        npz_data = np.load(data_path)
        if self.phase == 'train':
            self.data = npz_data['x_train']
            self.label = np.argmax(npz_data['y_train'], axis=-1)
        elif self.phase == 'eval':
            self.data = npz_data['x_test']
            self.label = np.argmax(npz_data['y_test'], axis=-1)
        else:
            raise NotImplementedError('data phase only supports train/test')
        nan_out = np.isnan(self.data.mean(-1).mean(-1))==False
        self.data = self.data[nan_out]
        self.label = self.label[nan_out]
        self.sample_name = [self.phase + '_' + str(i) for i in range(len(self.data))]
        N, T, _ = self.data.shape
        self.data = self.data.reshape((N, T, 2, 25, 3)).transpose(0, 4, 1, 3, 2) # N C T V M
        
            
    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        # (C, T, V, M)
        pose_data = np.array(self.data[idx])
        label = self.label[idx]
        
        pose_data = graph_processing(pose_data, self.graph, self.processing, no_changes=True)
        data_new = multi_input(pose_data, self.conn, self.inputs, self.center, no_changes=True)
        
        try:
            assert list(data_new.shape) == self.datashape
        except AssertionError:
            logging.info('data_new.shape: {}'.format(data_new.shape))
            raise ValueError()
        
        data_new, is_two_persons = self.ntu_data_preprocess(data_new, label)
        
        data_new = data_new.permute(1,3,2,0).contiguous() # (T, M, V, C)

        return data_new, label, is_two_persons
    

    def ntu_data_preprocess(self, data, label):
        """Script from InterAction code for loading samples"""
        
        # classes involving two persons
        two_persons = list(range(49,60)).extend(list(range(105,120)))
        if (label not in two_persons) and self.repeat:
            self.repeat_skeleton_for_one_body(data,label) # shape: M,T,V,n*C

        is_two_persons = (label in two_persons) if not self.repeat else True

        return data, is_two_persons



    def repeat_skeleton_for_one_body(self, data): # InterAction script
        """
            Our model requires that for actions involving one human,
            the second skeleton should be the same as the first skeleton.
            This is needed for the Interact module in our model.
        """
        # We do this for the action class involving just one human
        data[:,:,:,1] = data[:,:,:,0]

    
    def get_datashape(self):
        I = len(self.inputs) if self.inputs.isupper() else 1
        C = self.input_dims if self.inputs in [
            'joint', 'joint-motion', 'bone', 'bone-motion'] else self.input_dims*2
        T = self.max_frame
        V = self.num_node
        M = self.M // self.num_person
        return [I, C, T, V, M]
    

    