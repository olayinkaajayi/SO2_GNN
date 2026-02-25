import torch
import torch.nn as nn
import torch.nn.init as init
import numpy as np
import torch.nn.functional as F
from src.model.mlp import MLP

class SO2_GCN(torch.nn.Module):
    
    def __init__(self, in_dim, A, angle_partitions=4, rot_one_axis=False, num_heads=1, **kwargs):
        super(SO2_GCN, self).__init__()

        self.n = angle_partitions # partitions of interval
        self.rot_one_axis = rot_one_axis # rotate across multiple axis
        self.num_heads = num_heads

        N = A.shape[0]
        axes = ['y'] if rot_one_axis else ['x', 'y', 'z']
        self.axes = axes

        # t_k: shape [num_heads, n] per axis — separate angle params per head
        self.t_k = nn.ParameterDict({
            ax: nn.Parameter(torch.randn(num_heads, self.n))
            for ax in axes
        })

        # sigma_k: fused across heads using a single weight [num_heads, N*in_dim, n]
        # Equivalent to num_heads independent Linear(N*in_dim, n) layers
        self.sigma_k = nn.ParameterDict({
            ax: nn.Parameter(torch.randn(num_heads, N * in_dim, self.n) * (N * in_dim) ** -0.5)
            for ax in axes
        })
        self.sigma_bias = nn.ParameterDict({
            ax: nn.Parameter(torch.zeros(num_heads, self.n))
            for ax in axes
        })
        
        self.swap_wt_identity = False
        self.threshold = 0.01
        
        self.partition = True # decide if we partition the angles or just learn across the full range.
        self.strategy = 'circular-2' # options: 'default', 'circular', 'circular-2', 'tanh-1', 'tanh-2', 'sigmoid'


    def forward(self, x):
        """
            x shape: [N*M, T, V, C]          (num_heads=1, original behaviour)
            or [N*M, T, V, num_heads, C]     (multi-head mode)

            Returns same shape as input.
        """
        if self.num_heads == 1:
            # Keep original interface: squeeze/unsqueeze head dim internally
            x = x.unsqueeze(-2)          # [N*M, T, V, 1, C]

       # x: [N*M, T, V, H, C]
        if not self.rot_one_axis:
            Rx = x
            for ax in self.axes:
                Rx = self.R_t(Rx, axis=ax)
        else:
            Rx = self.R_t(x, axis='y')

        if self.num_heads == 1:
            Rx = Rx.squeeze(-2)          # back to [N*M, T, V, C]

        return Rx



    def R_t(self, x, axis='y'):
        """
            x shape: [N*M, T, V, H, C]  where H = num_heads, C = 3
        """
        NM, T, V, H, C = x.shape
        assert C == 3, "SO2_GCN expects 3D (x,y,z) input features"

        rotation_90_deg = self.rot_mat(axis=axis).to(x.device)   # [3, 3]
        zero_one_func = lambda a: F.sigmoid(a)

        # --- sigma: [N*M, T, H, n] ---
        # Flatten V and C into V*in_dim for each frame
        x_flat = x.permute(0, 1, 3, 2, 4).reshape(NM, T, H, V * C)  # [NM, T, H, V*C]
        # Batched linear per head: einsum over input dim
        # sigma_k[axis]: [H, V*C, n]
        sigma = torch.einsum('bthd, hdn -> bthn', x_flat, self.sigma_k[axis]) \
                + self.sigma_bias[axis]  # [NM, T, H, n]

        # --- learnt_t: [H, n] ---
        learnt_t = self.learn_angle_strategies(axis, device=x.device)  # [H, n]
        
        # Build rotation matrices: [H, n, 3, 3]
        # learnt_t controls the magnitude of each rotation basis
        exponent = rotation_90_deg.unsqueeze(0).unsqueeze(0) \
                   * learnt_t.unsqueeze(-1).unsqueeze(-1)    # [H, n, 3, 3]

        rot_mats = torch.matrix_exp(-exponent)               # [H, n, 3, 3]

        # --- Weighted sum of rotation matrices ---
        # z1_f: [NM, T, H, n, 1, 1]
        z1_f = zero_one_func(sigma).unsqueeze(-1).unsqueeze(-1)
        # rot_mats: [1, 1, H, n, 3, 3]
        rotate = z1_f * rot_mats.unsqueeze(0).unsqueeze(0)   # [NM, T, H, n, 3, 3]
        rotate_sum = rotate.sum(dim=-3)                       # [NM, T, H, 3, 3]

        # Note: Rotation matrices are pre-multiplied i.e. R.x, where x is a column vector:
        # TxHx3x3 . TxHx3xV --> TxHx3xV --(permute)--> TxVxHx3
        # Apply rotation: rotate_sum @ x^T → transpose back
        # x: [NM, T, H, V, C] → need [NM, T, H, C, V] for matmul
        x_mul = x.permute(0,1,3,4,2).contiguous()            # [NM, T, H, C, V]
        rot_x = rotate_sum.matmul(x_mul)       # [NM, T, H, C, V]

        return rot_x.permute(0,1,4,2,3).contiguous()         # [NM,T,V,H,C]


    
    def learn_angle_strategies(self, axis, device='cuda:0'):
        """Returns learnt angles of shape [H, n]"""

        if self.partition:
            modulus = torch.pi/self.n
            prev_mod = torch.tensor(
                [torch.pi * k / self.n for k in range(self.n)],
                device=device
            ).unsqueeze(0)  # [1, n] — broadcast over heads
        else:
            modulus = torch.pi
            prev_mod = 0
        
        t = self.t_k[axis]  # [H, n]

        if self.strategy == 'default':
            offset = t % modulus
        elif self.strategy == 'circular':
            offset = torch.atan2(torch.sin(t), torch.cos(t)) * (modulus / torch.pi)
        elif self.strategy == 'circular-2':
            offset = (0.5 * (torch.atan2(torch.sin(t), torch.cos(t)) + torch.pi)) * (modulus / torch.pi)
        elif self.strategy == 'tanh-1':
            offset = torch.tanh(t) * modulus
        elif self.strategy == 'tanh-2':
            offset = (torch.tanh(t) + 1) * 0.5 * modulus
        elif self.strategy == 'sigmoid':
            offset = torch.sigmoid(t) * modulus

        return prev_mod + offset  # [H, n]
    
    
    def rot_or_iden(self, zero_one, rot_mat):
        """Adapted for [NM, T, H, n] zero_one and [H, n, 3, 3] rot_mat."""
        NM, T, H, n = zero_one.shape
        rot_mat = rot_mat.unsqueeze(0).unsqueeze(0).expand(NM, T, H, n, 3, 3).clone()
        i, j, k, l = torch.where(zero_one < self.threshold)
        rot_mat[i, j, k, l] = torch.eye(3, device=zero_one.device)
        zero_one = zero_one.unsqueeze(-1).unsqueeze(-1)
        return zero_one * rot_mat


    
    def rot_mat(self, axis='y'):
        """Provides the rotation matrix as required."""
        # We can also use a scaling matrix, which would be useful 
        # for "in-the-wild" datasets e.g. Kinetics400 

        if axis == 'x':
            eta_yz = torch.tensor([[0,0,0], # R_x_90
                                    [0,0,-1],
                                    [0,1,0]], dtype=torch.float32)
            rot = eta_yz
        elif axis == 'z':
            eta_xy = torch.tensor([[0,-1,0], # R_z_90
                                    [1,0,0],
                                    [0,0,0]], dtype=torch.float32)
            rot = eta_xy

        else:
            eta_xz = torch.tensor([[0,0,-1], # R_y_90
                                    [0,0,0],
                                    [1,0,0]], dtype=torch.float32)
            rot = eta_xz

        return rot
    