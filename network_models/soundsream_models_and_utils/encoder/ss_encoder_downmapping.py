from torch import nn
from network_models.soundsream_models_and_utils.encoder.transformer import Encoder

class EncoderDownmapping(nn.Module):
    def __init__(self, embed_dim,  n_heads, ff_dim, n_layers, output, dropout = 0.1, max_seq_len=100):
        super(EncoderDownmapping, self).__init__()
        self.enc = Encoder(vocab_size=0, embed_dim=embed_dim, n_heads=n_heads, ff_dim=ff_dim, n_layers=n_layers, dropout=dropout, max_seq_len=max_seq_len)
        self.linear = nn.Linear(embed_dim*175, output)

    def forward(self, x, mask=None):
        out = x
        encOut = (self.enc(out, mask)).flatten(1)

        encOut = self.linear(encOut)
        return encOut
