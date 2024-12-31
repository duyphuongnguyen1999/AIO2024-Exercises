import torch
from Transformer import Transformer, TransformerEncoderCls

batch_size = 128
src_vocab_size = 1000
tgt_vocab_size = 2000
embed_dim = 200
max_length = 100
num_layers = 2
num_heads = 4
ff_dim = 256
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transformer = Transformer(
    src_vocab_size=src_vocab_size,
    tgt_vocab_size=tgt_vocab_size,
    embed_dim=embed_dim,
    max_length=max_length,
    num_layers=num_layers,
    num_heads=num_heads,
    ff_dim=ff_dim,
    device=device,
)
src = torch.randint(high=2, size=(batch_size, max_length))
tgt = torch.randint(high=2, size=(batch_size, max_length))


prediction = transformer(src, tgt)
print(prediction.shape)  # batch_size x max_length x tgt_vocab_size

transformer_encoder = TransformerEncoderCls(
    vocab_size=src_vocab_size,
    max_length=max_length,
    num_layers=num_layers,
    embed_dim=embed_dim,
    num_heads=num_heads,
    ff_dim=ff_dim,
    device=device,
)

prediction = transformer_encoder(src)
print(prediction.shape)
