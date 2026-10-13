import torch
from torch import nn
from ..utils import create_norm
import math

def select_pe_encoder(pe):
    if pe in ['sin', 'sinu', 'sinusoidal']:
        return Sinusoidal2dPE
    elif pe in ['learnable', 'bin']:
        return Learnable2dPE
    elif pe in ['naive', 'mlp']:
        return NaivePE
    elif pe in ['lap', 'graphlap', 'lappe']:
        return GraphLapPE
    elif pe in ['rff']:
        return RFFPE
    else:
        raise NotImplementedError(f'Unsupported positional encoding type: {pe}')

class RFFPE(nn.Module):
    def __init__(self, out_dim=512, num_feats=64, scales=[1.0, 5.0, 10.0]):
        super().__init__()
        # self.B = nn.Parameter(torch.cat([torch.randn(num_feats, 2) * s for s in scales], dim=0), requires_grad=False)
        B = torch.cat([torch.randn(num_feats, 2) * s for s in scales], dim=0)  # shape: [num_feats*len(scales), 2]
        self.register_buffer('B', B)
        self.proj = nn.Linear(len(scales) * num_feats * 2, out_dim)
        self.missing_pe = nn.Parameter(torch.randn(1, out_dim) * 1e-2)



    def forward(self, coords):
        if coords is None:
            return self.missing_pe.expand(1, -1)
        coords = self._normalize(coords)
        proj = 2 * math.pi * coords @ self.B.T
        emb = torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)
        return self.proj(emb)

    def _normalize(self, coords):
        coords_min = coords.min(dim=0, keepdim=True).values
        coords_max = coords.max(dim=0, keepdim=True).values
        coords_norm = (coords - coords_min) / (coords_max - coords_min + 1e-6)

        return coords_norm.to(self.B.dtype)
    
class Sinusoidal2dPE(nn.Module):
    def __init__(self, d_model, height=1000, width=1000):
        """
        :param d_model: dimension of the model
        :param height: height of the positions
        :param width: width of the positions
        """
        super().__init__()
        if d_model % 4 != 0:
            raise ValueError("Cannot use sin/cos positional encoding with "
                             "odd dimension (got dim={:d})".format(d_model))
        # import pdb;pdb.set_trace()
        self.d_model = d_model
        self.height = height
        self.width = width
        self.pe_key = 'coord'
        self.missing_pe = nn.Parameter(torch.randn(d_model) * 1e-2)

        pe = torch.zeros(d_model, height, width)
        # Each dimension use half of d_model
        d_model = int(d_model / 2)
        div_term = torch.exp(torch.arange(0., d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pos_w = torch.arange(0., width).unsqueeze(1)
        pos_h = torch.arange(0., height).unsqueeze(1)
        pe[0:d_model:2, :, :] = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
        pe[1:d_model:2, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
        pe[d_model::2, :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
        pe[d_model + 1::2, :, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
        self.pe_enc = nn.Embedding.from_pretrained(pe.flatten(1).T)

    def forward(self, coordinates):
        # import pdb;pdb.set_trace()
        if coordinates[0][0] == -1:
            return self.missing_pe.unsqueeze(0).expand(coordinates.shape[0], -1)
        x = coordinates[:, 0]
        y = coordinates[:, 1]
        x = ((x*1.02-0.01) * self.width).long()
        y = ((y*1.02-0.01) * self.height).long()
        x[x >= self.width] = self.width - 1
        y[y >= self.height] = self.height - 1
        x[x < 0] = 0
        y[y < 0] = 0
        pe_input = x * self.width + y
        return self.pe_enc(pe_input)

class Learnable2dPE(nn.Module):
    def __init__(self, d_model, height=100, width=100):
        """
        :param d_model: dimension of the model
        :param height: height of the positions
        :param width: width of the positions
        """
        super().__init__()
        self.pe_enc = nn.Embedding(height * width, d_model)
        self.missing_pe = nn.Parameter(torch.randn(d_model) * 1e-2)
        self.pe_key = 'coord'

    def forward(self, coordinates):
        if coordinates[0][0] == -1:
            return self.missing_pe.unsqueeze(0).expand(coordinates.shape[0], -1)
        x = coordinates[:, 0]
        y = coordinates[:, 1]
        x = ((x*1.02-0.01) * self.width).long()
        y = ((y*1.02-0.01) * self.height).long()
        x[x >= self.width] = self.width
        y[y >= self.height] = self.height
        x[x < 0] = 0
        y[y < 0] = 0
        pe_input = x * self.width + y
        return self.pe_enc(pe_input)

class NaivePE(nn.Module):
    def __init__(self, d_model, coord_dim = 2, height=None, width=None):
        """
        :param d_model: dimension of the model
        :param coord_dim: dimension of coordinates
        :param height: placeholder
        :param width: placeholder
        """
        super().__init__()
        self.pe_enc = nn.Sequential(
                            nn.Linear(coord_dim, d_model),
                            nn.PReLU(),
        )
        self.missing_pe = nn.Parameter(torch.randn(d_model) * 1e-2)
        self.pe_key = 'coord'

    def forward(self, coordinates):
        if coordinates[0][0] == -1:
            return self.missing_pe.unsqueeze(0).expand(coordinates.shape[0], -1)
        return self.pe_enc(coordinates)

class GraphLapPE(nn.Module):
    def __init__(self, d_model, k = 10, height=None, width=None):
        """
        :param d_model: dimension of the model
        :param k: top k
        :param height: placeholder
        :param width: placeholder
        """
        super().__init__()
        self.k = k
        self.pe_enc = nn.Sequential(
                            nn.Linear(k, d_model),
                            nn.PReLU(),
        )
        self.missing_pe = nn.Parameter(torch.randn(d_model) * 1e-2)
        self.pe_key = 'eigvec'

    def forward(self, eigvec):
        if eigvec[0][0] == -1:
            return self.missing_pe.unsqueeze(0).expand(eigvec.shape[0], -1)
        eigvec = eigvec * (torch.randint(0, 2, (self.k, ), dtype=torch.float, device=eigvec.device)[None, :]*2-1)
        return self.pe_enc(eigvec)

