import torch
import torch.nn as nn


class TokenAndPositionEmbedding(nn.Module):
    def __init__(self, vocab_size, embed_dims, max_length, device="gpu"):
        super().__init__()
        self.device = device
        self.word_embed = nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=embed_dims
        )

        self.pos_emb = nn.Embedding(num_embeddings=max_length, embedding_dim=embed_dims)

    def forward(self, x):
        N, seq_len = x.size()
        positions = torch.arange(0, seq_len).expand(N, seq_len).to(self.device)
        output1 = self.word_embed(x)
        output2 = self.pos_emb(positions)
        output = output1 + output2
        return output
