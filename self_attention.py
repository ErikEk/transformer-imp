import torch
import torch.nn as nn
from torch.nn import functional as F
B, T, C = 4, 8, 32 # batch, time, channels
x = torch.randn(B,T,C)

# lets see a single head perform self-attention
head_size = 16
key = nn.Linear(C, head_size, bias=False)
query = nn.Linear(C, head_size, bias=False)
value = nn.Linear(C, head_size, bias=False)
k = key(x)   # (B, T, 16)
q = query(x) # (B, T, 16)
wei = q @ k.transpose(-2,-1) # (B, T, 16) @ (B, 16, T) -> (B, T, T)

tril = torch.tril(torch.ones(T, T))

# This will be removed in a encoder block
wei = wei.masked_fill(tril == 0, float('-inf'))

wei = F.softmax(wei, dim=-1)
v = value(x)
out = wei @ v

print(wei[0])
#print(out[0])
print(out.shape)
