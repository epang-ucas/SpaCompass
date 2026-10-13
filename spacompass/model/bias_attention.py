import torch
import torch.nn as nn
import torch.nn.functional as F

class BiasTransformer(nn.Module):
    def __init__(self, dim, depth, heads, bias_layers=2, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList()
        self.alpha = nn.ParameterList()
        self.bias_layers = bias_layers  # Only first few layers apply spatial bias

        for i in range(depth):
            self.layers.append(
                nn.TransformerEncoderLayer(
                    d_model=dim,
                    nhead=heads,
                    dim_feedforward=dim,
                    dropout=dropout,
                    batch_first=True
                )
            )
            init_alpha = 0.1 if i < bias_layers else 0.0
            self.alpha.append(nn.Parameter(torch.tensor(init_alpha)))

    def forward(self, x, pos=None):
        """
        x: (B, N, D) spot embeddings
        pos: (B, N, 2) spot coordinates (um or pixel), will be normalized internally if not already
        """
        B, N, D = x.shape
        # import pdb;pdb.set_trace()

        if pos is not None:
            # Check if coordinates are already normalized to [0, 1]
            if pos.max() > 1.01 or pos.min() < -0.01:
                # Normalize coordinates to [0, 1] within each slice
                min_pos = pos.min(dim=1, keepdim=True)[0]  # (B, 1, 2)
                max_pos = pos.max(dim=1, keepdim=True)[0]
                pos = (pos - min_pos) / (max_pos - min_pos + 1e-8)  # (B, N, 2)

            dist = torch.cdist(pos, pos, p=2)  # (B, N, N)
            bias = -dist ** 2  # Gaussian spatial bias
            # import pdb;pdb.set_trace()
        else:
            bias = None

        for i, layer in enumerate(self.layers):
            if bias is not None and i < self.bias_layers:
                x = self._apply_attention_with_bias(layer, x, bias, alpha=self.alpha[i])
            else:
                x = layer(x)
        return x

    def _apply_attention_with_bias(self, layer, x, bias, alpha):
        attn = layer.self_attn
        B, N, D = x.shape
        # import pdb;pdb.set_trace()
        qkv = F.linear(x, attn.in_proj_weight, attn.in_proj_bias)  # (B, N, 3D)
        q, k, v = qkv.chunk(3, dim=-1)
    
        q = q.view(B, N, attn.num_heads, -1).transpose(1, 2)
        k = k.view(B, N, attn.num_heads, -1).transpose(1, 2)
        v = v.view(B, N, attn.num_heads, -1).transpose(1, 2)

        attn_logits = torch.matmul(q, k.transpose(-2, -1)) / (q.shape[-1] ** 0.5)
        bias = bias.unsqueeze(1)  # (B, 1, N, N)
        attn_logits = attn_logits + alpha * bias

        attn_weights = attn_logits.softmax(dim=-1)
        attn_output = torch.matmul(attn_weights, v)  # (B, H, N, D//H)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, N, D)
        attn_output = attn.out_proj(attn_output)
        
        # Residual + FFN
        x = x + layer.dropout1(attn_output)
        x = layer.norm1(x)
        x = x + layer.dropout2(layer.linear2(layer.dropout(layer.activation(layer.linear1(x)))))
        x = layer.norm2(x)
        return x
