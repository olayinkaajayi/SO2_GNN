import torch
import torch.nn as nn
import torch.nn.init as init
import numpy as np
import torch.nn.functional as F
from src.model.mlp import MLP

class SO2_GCN(torch.nn.Module):
    
    def __init__(self, in_dim, hidden_dim, A, angle_partitions=4, rot_one_axis=False, **kwargs):
        super(SO2_GCN, self).__init__()

        self.n = angle_partitions # partitions of interval
        self.rot_one_axis = rot_one_axis # rotate across multiple axis

        self.t_k = nn.ParameterDict({
            'x': nn.Parameter(torch.randn(self.n)),
            'y': nn.Parameter(torch.randn(self.n)),
            'z': nn.Parameter(torch.randn(self.n))
        }) if not rot_one_axis else nn.ParameterDict({
            'y': nn.Parameter(torch.randn(self.n))}) # should each lie in a disjoint set between [0,\pi)
        
        N = A.shape[0] # Number of nodes on skeleton graph
        self.sigma_k = nn.ModuleDict({ # consider reducing num_layers to 2
            'x': MLP(num_layers=3, input_dim=N*in_dim, hidden_dim=hidden_dim, output_dim=self.n),
            'y': MLP(num_layers=3, input_dim=N*in_dim, hidden_dim=hidden_dim, output_dim=self.n),
            'z': MLP(num_layers=3, input_dim=N*in_dim, hidden_dim=hidden_dim, output_dim=self.n)
        }) if not rot_one_axis else nn.ModuleDict({
            'y': MLP(num_layers=3, input_dim=N*in_dim, hidden_dim=hidden_dim, output_dim=self.n)}) # should each lie in the set [-1,+1] or [0,1] ??
        

    def forward(self, x):
        """This function implements the SO(2)-GCN model designed for different axis of rotations."""
        # x shape: N*M,T,V,C

        if not self.rot_one_axis:
            Rx = []
            ax = ['x','y','z']
            for i in range(len(ax)):
                Rx.append( self.R_t(x, axis=ax[i]) )
        else:
            Rx = self.R_t(x, axis='y')

        return Rx



    def R_t(self, x, axis='y'):
        """This function helps us achieve rotation equivariance."""
        # x shape: N*M,T,V,C
        NM, T, V, C = x.shape

        rotation_90_deg = self.rot_mat(axis=axis).to(x.device)
        
        # zero_one_func = lambda a: torch.exp(-(a**2)) # we want it to be close enough to 1 when the angle is relevant.
                                                    # Else it can push it to zeros as far as possible
        zero_one_func = lambda a: F.sigmoid(a) # This turned out to give a better result.

        sigma = self.sigma_k[axis](x.view(NM,T,-1).unsqueeze(2)).squeeze(-2) # shape: N*M,T,self.n --> unsqueeze and squeeze because of batchnorm shape in MLP.
        modulus = torch.pi/self.n
        prev_mod = torch.tensor(list(map(lambda k: torch.pi*(k)/self.n, range(self.n)))).to(x.device)
        learnt_t = prev_mod + (self.t_k[axis] % modulus) # ?? Would the modulus affect the differentiation (calculus) ??
        
        learnt_t = learnt_t.unsqueeze(-1).unsqueeze(-1).repeat(1,3,3) # shape: self.n,3,3
        exponent =  rotation_90_deg.unsqueeze(0) * learnt_t # rotation should be within [0,\pi)

        # repeat to distribute and allow for multiplication
        z1_f = zero_one_func(sigma).unsqueeze(-1).unsqueeze(-1).repeat(1,1,1,3,3) # shape: N*M,T,self.n,3,3

        rotate = z1_f*torch.matrix_exp(-exponent) # shape: N*M,T,self.n,3,3
                                                  # ?? is the "negative" relevant ?? It would just affect the direction of rotation.
        rotate_sum = rotate.sum(dim=-3) # shape: N*M,T,3,3

        # Note: Rotation matrices are pre-multiplied i.e. R.x, where x is a column vector:
        # Tx3x3 . Tx3xV --> Tx3xV --(transpose)--> TxVx3
        rot_x = rotate_sum.matmul(x.transpose(-1,-2)) # shape: N*M,T,C,V

        return rot_x.transpose(-1,-2) # shape: N*M,T,V,C

    
    def rot_mat(self, axis='y'):
        """Provides the rotation matrix as required."""
        # We can also use a scaling matrix, which would be useful 
        # for "in the wild" datasets e.g. Kinetics400 

        rot = torch.zeros([self.n,self.n])

        if axis == 'x':
            eta_yz = torch.tensor([[1,0,0], # R_x_90
                                    [0,0,-1],
                                    [0,1,0]])
            rot = eta_yz
        elif axis == 'z':
            eta_xy = torch.tensor([[0,-1,0], # R_z_90
                                    [1,0,0],
                                    [0,0,1]])
            rot = eta_xy

        else:
            eta_xz = torch.tensor([[0,0,-1], # R_y_90
                                    [0,1,0],
                                    [1,0,0]])
            rot = eta_xz

        return rot
    