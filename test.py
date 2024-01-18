import torch

# 假设的张量
bsz = 3
seq_len = 5
weight = torch.zeros(bsz, seq_len, 2)
weight_bias = torch.ones(1, 2)


# print(f"weight {weight} \n weight_bias {weight_bias}")

# 扩展weight_bias以匹配weight的形状
weight_bias_expanded = weight_bias.unsqueeze(1)  # 现在形状是(1, 1, 2)

# 由于weight的seq_len维度是大小不为1的维度，PyTorch将会自动扩展weight_bias
# 来匹配这个维度，执行广播相加
result = torch.sigmoid_(weight) + weight_bias_expanded
print(result)
