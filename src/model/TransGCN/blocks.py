import torch
from torch import nn
from src.model.TransGCN.so2_gnn import SO2_GCN
from src.model.mlp import MLP
import torch.nn.functional as F

class SO2_GCN_Block(nn.Module):
    def __init__(self, in_channels, out_channels, A, so2_arg, **kwargs):
        super(SO2_GCN_Block, self).__init__()

        self.heads = so2_arg['heads']
        angle_partitions = so2_arg['angle_partitions']
        rot_one_axis = so2_arg['rot_one_axis']
        self.in_channels = in_channels
        self.out_channels = out_channels
        o_dim = out_channels//self.heads
        o_dim = o_dim if o_dim >= 32 else 32 # set minimum head size to 32

        if self.in_channels > 3:
            self.proj = nn.ModuleList([nn.Linear(self.in_channels,3) for _ in range(self.heads)])

            self.gcn = nn.ModuleList( [
                                    SO2_GCN(in_dim=3, hidden_dim=o_dim, A=A, angle_partitions=angle_partitions,
                                            rot_one_axis=rot_one_axis, **kwargs)
                                            for _ in range(self.heads)]
                                    )

            # self.merge_heads = nn.ModuleDict( {'x': nn.Linear(self.heads*3, o_dim),
            #                                    'y': nn.Linear(self.heads*3, o_dim),
            #                                    'z': nn.Linear(self.heads*3, o_dim)})

            self.regroup = MLP(num_layers=1, input_dim=3*3*self.heads, # change to 3*o_dim for other case
                               hidden_dim=int(0.6*self.out_channels),
                               output_dim=self.out_channels)
        else:
            # consider setting rot_one_axis= True for the input feature i.e. in_dim=3
            self.gcn = SO2_GCN(in_dim=3,hidden_dim=self.out_channels, A=A, angle_partitions=angle_partitions, rot_one_axis=rot_one_axis, **kwargs)
            self.regroup = nn.Linear(3*3,self.out_channels)

        self.register_buffer('A', A) # Adjacency matrix

        # Add residual connection
        self.res_con = nn.Linear(in_channels, out_channels) if in_channels != out_channels else nn.Identity()


    def forward(self, x):
        # x [shape]: N*M,C,T,V
        x = x.permute(0,2,3,1).contiguous() # shape: N*M,T,V,C

        if self.in_channels > 3: # 3 because we have 3D space: x,y,z
            out = {'x':[], 'y':[], 'z':[]}
            # May be able to avoid for loop here by putting the heads into the linear layers
            for i in range(self.heads):
                y = self.proj[i](x)
                x_, y_, z_ = self.gcn[i](y)
                out['x'].append(x_); out['y'].append(y_); out['z'].append(z_)

            out = [torch.concatenate(out['x'], dim=-1).to(x.device),
                   torch.concatenate(out['y'], dim=-1).to(x.device),
                   torch.concatenate(out['z'], dim=-1).to(x.device)] # Each should have shape: N*M,T,V,C*self.head
            
            # out = [self.merge_heads['x'](torch.concatenate(out['x'], dim=-1).to(x.device)),
            #        self.merge_heads['y'](torch.concatenate(out['y'], dim=-1).to(x.device)),
            #        self.merge_heads['z'](torch.concatenate(out['z'], dim=-1).to(x.device))] # Each should have shape: N*M,T,V,o_dim
            
            # Another option is to pass the stacked tensors each through a separate linear layer,
            # then pass that output to be concatenated and weighed
            
            out = torch.concatenate(out, dim=-1).to(x.device) # shape: N*M,T,V,C*self.head*3
            out = self.get_normalized_adjacency(self.A, batch_size=x.size(0))@self.regroup(out) # shape: N*M,T,V,C
        else:
            out = self.gcn(x) # shape: N*M,T,V,C
            out = torch.concatenate(out, dim=-1).to(x.device)
            out = self.get_normalized_adjacency(self.A, batch_size=x.size(0))@self.regroup(out) # shape: N*M,T,V,C

        out = self.res_con(x) + F.relu(out)

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
            


class Spatial_Basic_Block(nn.Module):
    def __init__(self, in_channels, out_channels, max_graph_distance, A, edge_importance=True, adaptive=False):
        super(Spatial_Basic_Block, self).__init__()

        if in_channels == out_channels:
            self.residual = lambda x: x
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels),
            ) 

        self.conv = SpatialGraphConv(in_channels, out_channels, max_graph_distance)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        if adaptive:
            self.A = nn.Parameter(A[:max_graph_distance+1], requires_grad=True)
        else:
            self.register_buffer('A', A[:max_graph_distance+1])
        self.edge = nn.Parameter(torch.ones_like(A[:max_graph_distance+1]), requires_grad=edge_importance)


    def forward(self, x):

        res_block = self.residual(x)

        x = self.conv(x, self.A*self.edge)
        x = self.bn(x)
        x = self.relu(x + res_block)

        return x

class Temporal_Basic_Block(nn.Module):
    def __init__(self, channels, temporal_window_size, stride=1):
        super(Temporal_Basic_Block, self).__init__()

        padding = ((temporal_window_size - 1) // 2, 0)

        if stride == 1:
            self.residual = lambda x: x
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(channels, channels, 1, (stride,1)),
                nn.BatchNorm2d(channels),
            )

        self.conv = nn.Conv2d(channels, channels, (temporal_window_size,1), (stride,1), padding)
        self.bn = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, res_module):

        res_block = self.residual(x)

        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x + res_block + res_module)

        return x

class Temporal_MultiScale_Block(nn.Module):
    def __init__(self, out_channels, kernel_size=3, stride=1, dilations=[1,2], residual_kernel_size=1, **kwargs):

        super().__init__()
        in_channels = out_channels
        assert out_channels % (len(dilations) + 2) == 0, '# out channels should be multiples of # branches'

        # Multiple branches of temporal convolution
        self.num_branches = len(dilations) + 2
        branch_channels = out_channels // self.num_branches
        if type(kernel_size) == list:
            assert len(kernel_size) == len(dilations)
        else:
            kernel_size = [kernel_size]*len(dilations)
        # Temporal Convolution branches
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    branch_channels,
                    kernel_size=1,
                    padding=0),
                nn.BatchNorm2d(branch_channels),
                nn.ReLU(inplace=True),
                TemporalConv(
                    branch_channels,
                    branch_channels,
                    kernel_size=ks,
                    stride=stride,
                    dilation=dilation),
            )
            for ks, dilation in zip(kernel_size, dilations)
        ])

        # Additional Max & 1x1 branch
        # pad = (3 + (3-1) * (1-1) - 1) // 2
        self.branches.append(nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True),
            # nn.Conv2d(branch_channels, branch_channels, kernel_size=(3, 1), padding=(pad,0), stride=(stride,1)),
            nn.MaxPool2d(kernel_size=(3,1), stride=(stride,1), padding=(1,0)),
            nn.BatchNorm2d(branch_channels)
        ))

        self.branches.append(nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=1, padding=0, stride=(stride,1)),
            nn.BatchNorm2d(branch_channels)
        ))

        # Residual connection
        if (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = TemporalConv(in_channels, out_channels, kernel_size=residual_kernel_size, stride=stride)

    def forward(self, x):
        # Input dim: (N,C,T,V)
        res = self.residual(x)
        branch_outs = []
        for tempconv in self.branches:
            out = tempconv(x)
            branch_outs.append(out)

        out = torch.cat(branch_outs, dim=1)
        out += res 
        return out


class ST_Person_Attention(nn.Module):
    def __init__(self, channel, parts, reduct_ratio, bias=True, **kwargs):
        super(ST_Person_Attention, self).__init__()

        self.parts = parts
        self.joints = nn.Parameter(self.get_corr_joints(), requires_grad=False)
        self.mat = nn.Parameter(self.get_mean_matrix(), requires_grad=False)
        inner_channel = channel // reduct_ratio

        self.fcn = nn.Sequential(
            nn.Conv2d(channel, inner_channel, kernel_size=1, bias=bias),
            nn.BatchNorm2d(inner_channel),
            nn.ReLU(),
        )
        self.conv_t = nn.Conv2d(inner_channel, channel, kernel_size=1)
        self.conv_v = nn.Conv2d(inner_channel, channel, kernel_size=1)
    
        self.bn = nn.BatchNorm2d(channel)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        N, C, T, V = x.size()
        P = len(self.parts)
        res = x

        x_t = x.mean(3, keepdims=True) # N,C,T,1
        # x_v = x.mean(2, keepdims=True).transpose(2, 3) # N,C,V,1
        x_p = (x.mean(2, keepdims=True) @ self.mat).transpose(2, 3) # N,C,P,1
        x_att = self.fcn(torch.cat([x_t, x_p], dim=2)) # N,C,(T+P),1
        x_t, x_p = torch.split(x_att, [T, P], dim=2) 
        x_t_att = self.conv_t(x_t).sigmoid() # N,C,T,1
        
        x_p_att = self.conv_v(x_p.transpose(2, 3)).sigmoid() # N,C,1,P
        x_v_att = x_p_att.index_select(3, self.joints) # N,C,1,V

        x_att = x_t_att * x_v_att
        return self.act(self.bn(x * x_att) + res)
    
    def get_corr_joints(self):
        num_joints = sum([len(part) for part in self.parts])
        joints = [j for i in range(num_joints) for j in range(len(self.parts)) if i in self.parts[j]]
        return torch.LongTensor(joints)
    
    def get_mean_matrix(self):
        num_joints = sum([len(part) for part in self.parts])
        Q = torch.zeros(num_joints, len(self.parts))
        for j in range(len(self.parts)):
            n = len(self.parts[j])
            for joint in self.parts[j]:
                Q[joint][j] = 1.0/n
        return Q
    
    
# Thanks to YAN Sijie for the released code on Github (https://github.com/yysijie/st-gcn)
class SpatialGraphConv(nn.Module):
    def __init__(self, in_channels, out_channels, max_graph_distance):
        super(SpatialGraphConv, self).__init__()

        # spatial class number (distance = 0 for class 0, distance = 1 for class 1, ...)
        self.s_kernel_size = max_graph_distance + 1

        # weights of different spatial classes
        self.gcn = nn.Conv2d(in_channels, out_channels*self.s_kernel_size, 1)

    def forward(self, x, A):

        # numbers in same class have same weight
        x = self.gcn(x)

        # divide nodes into different classes
        n, kc, t, v = x.size()
        x = x.view(n, self.s_kernel_size, kc//self.s_kernel_size, t, v)

        # spatial graph convolution
        x = torch.einsum('nkctv,kvw->nctw', (x, A[:self.s_kernel_size])).contiguous()

        return x
    
class TemporalConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1):
        super(TemporalConv, self).__init__()
        pad = (kernel_size + (kernel_size-1) * (dilation-1) - 1) // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, 1),
            padding=(pad, 0),
            stride=(stride, 1),
            dilation=(dilation, 1))

        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x