# IMPLEMENTATION found here: https://blog.floydhub.com/the-transformer-in-pytorch/ + https://www.kaggle.com/code/arunmohan003/transformer-from-scratch-using-pytorch, https://github.com/SamLynnEvans/Transformer/blob/master/Models.py
import torch.nn as nn
import torch
import torch.nn.functional as F

class Embedding(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        """
        Args:
            vocab_size: size of vocabulary
            embed_dim: dimension of embeddings
        """
        super(Embedding, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
    def forward(self, x):
        """
        Args:
            x: input vector
        Returns:
            out: embedding vector
        """
        out = self.embed(x)
        return out

class PositionalEmbedding(nn.Module):
    def __init__(self, max_seq_len, embed_dim):
        """
        Args:
            max_seq_len: maximum sequence length
            embed_dim: dimension of embeddings
        """
        super(PositionalEmbedding, self).__init__()
        # We implement a learned positional embedding
        self.pos_embed = nn.Embedding(max_seq_len, embed_dim)

    def forward(self, x):
        """
        Args:
            x: input vector
        Returns:
            out: embedding vector
        """
        # get sequence length
        seq_len = x.size(1)
        # create position vector => [0, 1, 2, ..., seq_len]
        pos = torch.arange(0, seq_len).unsqueeze(0).to(x.device)
        # create embedding vector
        out = self.pos_embed(pos)
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, embed_dim, dropout = 0.1):
        """
        Args:
            n_heads: number of attention heads
            embed_dim: dimension of embeddings
            dropout: dropout rate
        """
        super(MultiHeadAttention, self).__init__()
        # define number of heads, embedding dimension and single head dimension
        self.n_heads = n_heads
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // n_heads
        assert self.head_dim * n_heads == embed_dim, "Embedding dimension must be divisible by number of heads"
        # define query, key, value matrices
        self.query_matrix = nn.Linear(embed_dim, embed_dim)
        self.key_matrix = nn.Linear(embed_dim, embed_dim)
        self.value_matrix = nn.Linear(embed_dim, embed_dim)
        # define dropout layer
        self.dropout = nn.Dropout(dropout)
        # define output layer
        self.out = nn.Linear(embed_dim, embed_dim)

    def forward(self, query, key, value, mask = None):
        # get batch size
        batch_size = query.size(0)  
        # perform linear operation/matrix multiplication and split into h heads
        key = self.key_matrix(key).view(batch_size, -1, self.n_heads, self.head_dim)
        query = self.query_matrix(query).view(batch_size, -1, self.n_heads, self.head_dim)
        value = self.value_matrix(value).view(batch_size, -1, self.n_heads, self.head_dim)
        # transpose to get dimensions batch_size * n_heads * seq_len * embed_dim
        key = key.transpose(1, 2)
        query = query.transpose(1, 2)
        value = value.transpose(1, 2)
        # calculate attention 
        scores = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim))
        # apply mask
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9) # mask == 0 means input is masked out, so we set the score to -1e9 (= -infinity), so that the softmax function will output 0 for this input which means no attention will be applied to this input
        # apply softmax to get attention weights
        scores = F.softmax(scores, dim = -1)
        # apply dropout
        if self.dropout is not None:
            scores = self.dropout(scores)
        # calculate attention scores
        scores = torch.matmul(scores, value)
        # concatenate heads 
        concat = scores.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)
        # pass through output layer
        out = self.out(concat)
        # return output 
        return out

class FeedForwardNetwork(nn.Module):
    def __init__(self, embed_dim, ff_dim=2048, dropout = 0.1):
        """
        Args:
            embed_dim: dimension of embeddings
            ff_dim: dimension of feedforward network
            dropout: dropout rate
        """
        super(FeedForwardNetwork, self).__init__()
        # define feedforward network
        self.linear_1 = nn.Linear(embed_dim, ff_dim)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(ff_dim, embed_dim)

    def forward(self, x):
        # pass through first linear layer
        out = self.linear_1(x)
        # pass through activation function
        out = F.relu(out)
        # pass through dropout
        out = self.dropout(out)
        # pass through second linear layer
        out = self.linear_2(out)
        # return output
        return out

class Norm(nn.Module):
    def __init__(self, embed_dim, eps = 1e-6):
        """
        Args:
            embed_dim: dimension of embeddings
            eps: epsilon
        """
        super(Norm, self).__init__()
        # define size of embeddings
        self.size = embed_dim
        # create two learnable parameters to calibrate normalisation
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))
        # define epsilon
        self.eps = eps
    
    def forward(self, x):
        # calculate mean and standard deviation
        mean = x.mean(-1, keepdim = True)
        std = x.std(-1, keepdim = True)
        # apply normalisation
        out = (x - mean) / (std + self.eps)
        # scale and shift
        out = self.alpha * out + self.bias
        # return output
        return out

class EncoderLayer(nn.Module):
    def __init__(self, embed_dim, n_heads, ff_dim, dropout = 0.1):
        """
        Encoder layer with one multi-head attention layer and one # feed-forward layer
        Args:
            embed_dim: dimension of embeddings
            n_heads: number of heads
            ff_dim: dimension of feedforward network
            dropout: dropout rate
        """
        super(EncoderLayer, self).__init__()
        # define sublayers
        self.norm_1 = Norm(embed_dim)
        self.norm_2 = Norm(embed_dim)
        self.attn = MultiHeadAttention(n_heads, embed_dim, dropout)
        self.ff = FeedForwardNetwork(embed_dim, ff_dim, dropout)
        # define dropout
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x, mask = None):
        # pass through first sublayer
        _x = self.norm_1(x)
        # define query, key and value
        query = key = value = _x
        # pass through multi-head attention layer
        out = self.attn(query, key, value, mask)
        # apply dropout
        out = self.dropout_1(out)
        # add residual connection
        out = x + out
        # pass through second sublayer
        _out = self.norm_2(out)
        # pass through feedforward network
        out = self.ff(_out)
        # apply dropout
        out = self.dropout_2(out)
        # add residual connection
        out = out + _out
        # return output
        return out


class DecoderLayer(nn.Module):
    def __init__(self, embed_dim, n_heads, ff_dim, dropout = 0.1):
        """
        Decoder layer with two multi-head attention layers and one feed-forward layer
        Args:
            embed_dim: dimension of embeddings
            n_heads: number of heads
            ff_dim: dimension of feedforward network
            dropout: dropout rate
        """
        super(DecoderLayer, self).__init__()
        # define sublayers
        self.norm_1 = Norm(embed_dim)
        self.norm_2 = Norm(embed_dim)
        self.norm_3 = Norm(embed_dim)
        # define multi-head attention layers
        self.attn_1 = MultiHeadAttention(n_heads, embed_dim, dropout)
        self.attn_2 = MultiHeadAttention(n_heads, embed_dim, dropout)
        # define feedforward network
        self.ff = FeedForwardNetwork(embed_dim, ff_dim, dropout)
        # define dropout
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.dropout_3 = nn.Dropout(dropout)

    def forward(self, x, encoder_outputs, src_mask = None, trg_mask = None):
        # pass through first sublayer
        _x = self.norm_1(x)
        # define query, key and value
        query = key = value = _x
        # pass through masked multi-head self-attention sublayer
        out = self.attn_1(query, key, value, trg_mask)
        # apply dropout
        out = self.dropout_1(out)
        # add residual connection
        out = out + x
        # pass through second sublayer
        _out = self.norm_2(out)
        # define query -> key and value -> encoder_outputs
        query = _out
        # pass through multi-head encoder-decoder attention sublayer
        out = self.attn_2(query, encoder_outputs, encoder_outputs, src_mask)
        # apply dropout
        out = self.dropout_2(out)
        # add residual connection
        out = out + _out
        # pass through third sublayer
        _out = self.norm_3(out)
        # pass through feedforward network
        out = self.ff(_out)
        # apply dropout
        out = self.dropout_3(out)
        # add residual connection
        out = out + _out
        # return output
        return out


class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, n_heads, ff_dim, n_layers, dropout = 0.1, max_seq_len=100):
        """
        Encoder with n_layers encoder layers
        Args:
            vocab_size: size of vocabulary
            embed_dim: dimension of embeddings
            n_heads: number of heads
            ff_dim: dimension of feedforward network
            n_layers: number of encoder layers
            dropout: dropout rate
        """
        super(Encoder, self).__init__()
        # define embedding layer -> NOT NEEDED -> we use our symbol embedding layer instead!
        # self.embed = nn.Embedding(vocab_size, embed_dim)
        # define positional encoding
        # self.positional_embedding = PositionalEmbedding(max_seq_len, embed_dim)
        # define encoder layers
        self.layers = nn.ModuleList([EncoderLayer(embed_dim, n_heads, ff_dim, dropout) for _ in range(n_layers)])
        # define normalisation
        self.norm = Norm(embed_dim)

    def forward(self, x, mask = None):
        # get embedding
        #out = self.embed(x)
        # add positional embedding
        #out = out + self.positional_embedding(x)
        out = x # ADDED -> TODO: refactor code to make the embedding optional!!
        # pass through encoder layers
        for layer in self.layers:
            out = layer(out, mask)
        # pass through normalisation
        out = self.norm(out)
        # return output
        return out

class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, n_heads, ff_dim, n_layers, dropout = 0.1, max_seq_len=100):
        """
        Decoder with n_layers decoder layers
        Args:
            vocab_size: size of vocabulary
            embed_dim: dimension of embeddings
            n_heads: number of heads
            ff_dim: dimension of feedforward network
            n_layers: number of decoder layers
            dropout: dropout rate
        """
        super(Decoder, self).__init__()
        # define embedding layer -> NOT NEEDED -> we use our symbol embedding layer instead!
        # self.embed = nn.Embedding(vocab_size, embed_dim)
        # define positional encoding
        # self.positional_embedding = PositionalEmbedding(max_seq_len, embed_dim)
        # define decoder layers
        self.layers = nn.ModuleList([DecoderLayer(embed_dim, n_heads, ff_dim, dropout) for _ in range(n_layers)])
        # define normalisation
        self.norm = Norm(embed_dim)

    def forward(self, x, encoder_outputs, gold_standard_rule_global_rule_index, src_mask = None, trg_mask = None):
        # get embedding
        # out = self.embed(x)
        # add positional embedding
        # out = out + self.positional_embedding(x)
        out = x # ADDED -> TODO: refactor code to make the embedding optional!!
        # pass through decoder layers
        for layer in self.layers:
            out = layer(out, encoder_outputs, src_mask, trg_mask)
        # pass through normalisation
        out = self.norm(out)
        # return output
        return out

class Transformer(nn.Module):
    def __init__(self, src_vocab_size, trg_vocab_size, embed_dim, n_heads, ff_dim, n_layers, dropout = 0.1):
        """
        Transformer model
        Args:
            src_vocab_size: size of source vocabulary
            trg_vocab_size: size of target vocabulary
            embed_dim: dimension of embeddings
            n_heads: number of heads
            ff_dim: dimension of feedforward network
            n_layers: number of layers
            dropout: dropout rate
        """
        super(Transformer, self).__init__()
        # define encoder
        self.encoder = Encoder(src_vocab_size, embed_dim, n_heads, ff_dim, n_layers, dropout)
        # define decoder
        self.decoder = Decoder(trg_vocab_size, embed_dim, n_heads, ff_dim, n_layers, dropout)
        # define output layer
        self.out = nn.Linear(embed_dim, trg_vocab_size)

    def create_src_mask(self, src, pad_idx = 1):
        """
        Args:
            src: source sequence
            pad_idx: index of padding token
        Returns:
            src_mask: mask for source sequence
        """
        # mask padding tokens
        src_mask = (src != pad_idx).unsqueeze(1).unsqueeze(2) # unsqueeze to add batch and head dimensions, alternatively unsqueeze(-2) ???
        # return mask
        return src_mask

    def create_trg_mask(self, trg):
        # TODO: this code is not working, fix it
        """
        Args:
            trg: target sequence
        Returns:
            trg_mask: mask for target sequence
        """
        # get batch size and target sequence length
        batch_size, trg_len = trg.shape
        # create a causal mask/subsequent mask, i.e. mask out future tokens
        trg_mask = torch.tril(torch.ones((trg_len, trg_len))).expand(
            batch_size, 1, trg_len, trg_len
        )
        # return mask
        return trg_mask

    def forward(self, src, trg, src_mask = None, trg_mask = None):
        # pass through encoder
        encoder_outputs = self.encoder(src, src_mask)
        # pass through decoder
        decoder_outputs = self.decoder(trg, encoder_outputs, src_mask, trg_mask)
        # pass through output layer
        out = self.out(decoder_outputs)
        # softmax over output vocabulary
        out = F.softmax(out, dim = -1)
        # return output
        return out