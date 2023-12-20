from .lora import LowRankLinear
import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import bmtrain as bmt

@torch.jit.script
def rms_layernorm(hidden: torch.Tensor, weight: torch.Tensor, eps: float):
    old_dtype = hidden.dtype
    variance = hidden.to(torch.float32).pow(2).mean(dim=-1, keepdim=True)
    hidden = (hidden * torch.rsqrt(variance + eps)).to(old_dtype)
    return hidden * weight

class LayerNorm(bmt.DistributedModule):
    """RMS LayerNorm"""

    def __init__(
        self,
        dim_norm: int,
        dtype: torch.dtype = torch.half,
        eps: float = 1e-5,
        init_var: float = 1.0,
    ):
        super().__init__()

        self.eps = eps
        self.dim_norm = dim_norm
        self.weight = nn.Parameter(torch.full((dim_norm,), init_var, dtype=dtype))

    def forward(self, x: torch.Tensor):
        """
        Args:
            x (:obj:`torch.Tensor` of shape ``(batch_size, seq_len, dim_norm)``): Input tensor that need to be normalized.
        Return:
            :obj:`torch.Tensor` of shape ``(batch_size, seq_len, dim_norm)``: The layernorm output.
        """  # noqa: E501
        assert x.size(-1) == self.dim_norm
        return rms_layernorm(x, self.weight, self.eps)

class attention_gate(bmt.DistributedModule):
    def __init__(self, output_size, gate_lora_r = 8, gate_lora_alpha = 8, gate_lora_dropout = 0.05):
        super().__init__()
        self.gate_lora_r = gate_lora_r
        self.gate_lora_alpha = gate_lora_alpha
        self.gate_lora_dropout = gate_lora_dropout

        self.gate_layernorm = LayerNorm(output_size)
        self.attention_gate_Q = LowRankLinear(output_size, output_size, r=gate_lora_r, lora_alpha=gate_lora_alpha, lora_dropout=gate_lora_dropout)
        self.attention_gate_K = LowRankLinear(output_size, output_size, r=gate_lora_r, lora_alpha=gate_lora_alpha, lora_dropout=gate_lora_dropout)
        ##去掉gate_V, 相当于以avg作为训练起点
        # self.attention_gate_V = LowRankLinear(output_size, output_size, r=gate_lora_r, lora_alpha=gate_lora_alpha, lora_dropout=gate_lora_dropout)

    def forward(self, hidden, delta_hiddens):

        delta_hiddens = delta_hiddens.detach()
        hidden = self.gate_layernorm(hidden)
        delta_hiddens = [self.gate_layernorm(delta_hidden) for delta_hidden in delta_hiddens.values()]
        attention_Q = self.attention_gate_Q(hidden)
        attention_keys = torch.stack([self.attention_gate_K(delta_hidden) for delta_hidden in delta_hiddens], dim=-1)  # (batch_size, 4096, 4096, 2)
        attention_values = torch.stack([delta_hidden for delta_hidden in delta_hiddens], dim=-1)  # (batch_size, 4096, 4096, 2)

        # 计算注意力分数
        # 首先，扩展 attention_Q 的维度以匹配 attention_keys
        attention_Q_expanded = attention_Q.unsqueeze(-1)  # (batch_size, 4096, 4096, 1)

        attention_scores = torch.matmul(attention_Q_expanded.transpose(-2, -1), attention_keys)  # (batch_size, 4096, 4096, 2)

        # 计算注意力权重
        attention_weights = F.softmax(attention_scores, dim=-1)  # (batch_size, 4096, 4096, 2)

        # 计算最终的注意力输出
        #attention_output = torch.matmul(attention_weights, attention_values.transpose(-2, -1))
        attention_output = torch.matmul(attention_values,attention_weights.transpose(-2, -1))  # (batch_size, 4096, 4096, 2)
        attention_output = attention_output.sum(dim=-1)  # (batch_size, 4096, 4096)

        # import pdb
        # pdb.set_trace()
        return attention_output