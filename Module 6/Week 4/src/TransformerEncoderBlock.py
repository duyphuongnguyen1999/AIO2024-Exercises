import torch.nn as nn

class TransformerEncoderBlock(nn.Module):
    def __init__(self, embed_dims, num_heads, ff_dim, dropout=0.1):