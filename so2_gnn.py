import torch
import torch.nn as nn
import numpy as np

class SO2_GNN(torch.nn.Module):
    
    def __init__(self, n=4, rot_one_axis=False):
        super().__init__()

        self.n = n # partitions of interval
        self.t_k = nn.Parameter(torch.randn(self.n)) # should each lie in a disjoint set between [0,\pi)
        self.sigma_k = nn.Parameter(torch.randn(self.n))

    def forward(self):

        pass

    def R_t(self, x, axis='y'):
        """This function helps us achieve rotation equivariance."""
        rotation_90_deg = self.rot_mat(axis=axis)
        
        zero_one_func = lambda a: torch.exp(-(a**2)) # we want it to be close enough to 1 when the angle is relevant.
                                                    # Else it can push it to zeros as far as possible
        
        for k in range(self.n):
            modulus = np.pi/self.n
            prev_mod = np.pi*(k)/self.n
            learnt_t = prev_mod + (self.t_k[k] % modulus) # ?? Would the modulus affect the differentiation (calculus) ??
            exponent =  rotation_90_deg * learnt_t # rotation should be within [0,\pi)

            rotate = zero_one_func(self.sigma_k[k])*torch.matrix_exp(-exponent) # ?? is the "negative" relevant ?? It would just affect the direction of rotation. 

        return rotate

    
    def rot_mat(self, axis='y'):
        """Provides the rotation matrix as required."""

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
        