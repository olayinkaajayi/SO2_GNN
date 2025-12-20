import torch
import torch.nn as nn
import torch.nn.init as init
import numpy as np
import torch.nn.functional as F

class GCN(torch.nn.Module):
    
    def __init__(self, in_dim, hidden_dim, A, **kwargs):
        super(GCN, self).__init__()

        self.weight_mat = nn.Linear(in_dim, hidden_dim, bias=False)

        init.xavier_uniform_(self.weight_mat.weight, gain=init.calculate_gain('relu')) # Use relu gain if ReLU follows

        self.register_buffer('A', A) # Adjacency matrix


    def forward(self, x):
        """This function implements the SO(2)-GCN model designed for different axis of rotations."""
        # x [shape]: N*M,C,T,V
        x = x.permute(0,2,3,1).contiguous() # shape: N*M,T,V,C


        D_A_DxW = self.get_normalized_adjacency(self.A, batch_size=x.size(0)).matmul(self.weight_mat(x))

        out = F.relu(D_A_DxW)

        out = out.permute(0,3,1,2).contiguous() # shape: N*M,C,T,V

        return out



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
        A_tilde = A + I
        
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


