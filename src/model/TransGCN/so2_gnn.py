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
        
        self.weight_mat = nn.ParameterList([nn.Linear(in_dim, hidden_dim, bias=False)
                                            for _ in range(3)]) if not rot_one_axis else nn.Linear(in_dim, hidden_dim, bias=False)

        for m in self.weight_mat:
            init.xavier_uniform_(m.weight, gain=init.calculate_gain('relu')) # Use relu gain if ReLU follows

        self.register_buffer('A', A) # Adjacency matrix


    def forward(self, x):
        """This function implements the SO(2)-GCN model designed for different axis of rotations."""
        # x shape: N*M,T,V,C

        RxW = 0
        if not self.rot_one_axis:
            ax = ['x','y','z']
            for i in range(len(ax)):
                RxW += self.weight_mat[i](self.R_t(x, axis=ax[i]))
        else:
            RxW = self.weight_mat(self.R_t(x, axis='y'))

        D_A_D_RxW = self.get_normalized_adjacency(self.A, batch_size=x.size(0))@RxW

        out = F.relu(D_A_D_RxW)

        return out



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
    


    def get_normalized_adjacency(self, A: torch.Tensor, batch_size) -> torch.Tensor:
        """
        Calculates the symmetrically normalized adjacency matrix (D^(-1/2) * A * D^(-1/2))
        after adding self-loops to A.

        Args:
            A: The input adjacency matrix (N x N) as a PyTorch Tensor.

        Returns:
            The normalized adjacency matrix (N x N) Tensor.
        """
        
        # Add Self-Loops (Calculate A_tilde = A + I)
        N = A.size(0)
        I = torch.eye(N, dtype=A.dtype, device=A.device)
#########################################################################
        A_tilde = A + I # consider adding a parameter to multiply I (as recommended in GCN paper) ####################
        
        # Calculate Degree Matrix D_tilde
        # Keep the dimension [:, 1] for broadcasting.
        D_tilde_diag = torch.sum(A_tilde, dim=1) # Shape: (N)
        
        # Calculate D_tilde^(-1/2)
        # A small epsilon is added for numerical stability in case of zero degrees.
        epsilon = 1e-12
        D_tilde_inv_sqrt = torch.pow(D_tilde_diag + epsilon, -0.5) # Shape: (N)
        
        # Convert to a diagonal matrix for matrix multiplication
        # D_tilde_inv_sqrt_matrix = torch.diag(D_tilde_inv_sqrt)
        
        # Calculate D^(-1/2) * A_tilde * D^(-1/2)
        
        # D_tilde_inv_sqrt needs to be reshaped to (N, 1) for multiplication along columns (rows of A_tilde).
        A_prime = A_tilde * D_tilde_inv_sqrt.unsqueeze(1)
        
        # Right multiplication: A_prime * D^(-1/2)
        # D_tilde_inv_sqrt (N) naturally broadcasts against the columns of A_prime (N x N).
        A_normalized = A_prime * D_tilde_inv_sqrt.unsqueeze(0)
        
        return A_normalized.unsqueeze(0).unsqueeze(0).repeat(batch_size,1,1,1) # batch_size x 1 x N x N


