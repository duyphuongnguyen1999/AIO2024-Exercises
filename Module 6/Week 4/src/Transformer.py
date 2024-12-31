import torch
import torch.nn as nn
from Encoder import TransformerEncoder
from Decoder import TransformerDecoder


class Transformer(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        tgt_vocab_size,
        embed_dim,
        max_length,
        num_layers,
        num_heads,
        ff_dim,
        dropout=0.1,
        device="cpu",
    ):
        super().__init__()
        self.device = device
        self.encoder = TransformerEncoder(
            src_vocab_size=src_vocab_size,
            embed_dim=embed_dim,
            max_length=max_length,
            num_heads=num_layers,
            num_heads=num_heads,
            ff_dim=ff_dim,
        )
        self.decoder = TransformerDecoder(
            tgt_vocab_size=tgt_vocab_size,
            embed_dim=embed_dim,
            max_length=max_length,
            num_layers=num_layers,
            num_heads=num_heads,
            ff_dim=ff_dim,
        )
        self.fc = nn.Linear(embed_dim, tgt_vocab_size)

    def generate_mask(self, src, tgt):
        src_seq_len = src.shape[1]
        tgt_seq_len = tgt.shape[1]

        src_mask = torch.zeros((src_seq_len, src_seq_len), device=self.device)
        src_mask = src_mask.type(torch.bool)

        tgt_mask = torch.ones((tgt_seq_len, tgt_seq_len), device=self.device)
        tgt_mask = (torch.triu(tgt_mask) == 1).transpose(0, 1)
        tgt_mask = (
            tgt_mask.float()
            .masked_fill(tgt_mask == 0, float("inf"))
            .masked_fill(tgt_mask == 1, float(0.0))
        )
        return src_mask, tgt_mask

    def forward(self, src, tgt):
        src_mask, tgt_mask = self.generate_mask(src, tgt)
        enc_output = self.encoder(src)
        dec_output = self.decoder(tgt, enc_output, src_mask, tgt_mask)
        output = self.fc(dec_output)
        return output
