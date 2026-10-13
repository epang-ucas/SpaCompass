import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphAttentionLayer(nn.Module):
    def __init__(self, in_dim, out_dim, heads=4, dropout=0.1):
        super().__init__()
        self.heads = heads
        self.out_dim = out_dim

        self.W = nn.Linear(in_dim, heads * out_dim, bias=False)
        self.a = nn.Parameter(torch.empty(size=(heads, 2 * out_dim)))
        nn.init.xavier_uniform_(self.a.data)

        self.leaky_relu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout)
        self.output_proj = nn.Linear(heads * out_dim, in_dim)

    def forward(self, x, adj):
        """
        x: (B, N, D) spot embedding
        adj: (B, N, N) bool mask for edges (邻接图)
        """
        B, N, _ = x.size()

        h = self.W(x)  # (B, N, H * D_out)
        h = h.view(B, N, self.heads, self.out_dim)  # (B, N, H, D_out)
        h_i = h.unsqueeze(2).expand(B, N, N, self.heads, self.out_dim)
        h_j = h.unsqueeze(1).expand(B, N, N, self.heads, self.out_dim)
        a_input = torch.cat([h_i, h_j], dim=-1)  # (B, N, N, H, 2 * D_out)

        e = self.leaky_relu((a_input * self.a).sum(dim=-1))  # (B, N, N, H)
        # import pdb;pdb.set_trace()
        # mask with adjacency matrix
        e = e.masked_fill(~adj.unsqueeze(-1), float("-inf"))
        e = e.clamp(min=-1e9, max=1e9)
        attn = F.softmax(e, dim=2)  # over neighbor dim (N)
        attn = self.dropout(attn)

        out = (attn.unsqueeze(-1) * h_j).sum(dim=2)  # sum over neighbors
        out = out.view(B, N, -1)
        return self.output_proj(out)


class GraphAttentionEncoder(nn.Module):
    def __init__(self, dim, layers=3, heads=4, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            GraphAttentionLayer(dim, dim // heads, heads=heads, dropout=dropout)
            for _ in range(layers)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(dim) for _ in range(layers)])

    def forward(self, x, adj):
        # import pdb;pdb.set_trace()
        adj = adj | torch.eye(adj.shape[1], dtype=torch.bool, device=adj.device).unsqueeze(0)
        for layer, norm in zip(self.layers, self.norms):
            x = norm(x + layer(x, adj))
        return x