import torch.nn as nn
from torch.nn.modules.transformer import TransformerEncoder, TransformerEncoderLayer
from torch.nn.modules import LayerNorm
from .embedding import BERTEmbedding


class BERT(nn.Module):
    def __init__(self, num_features, hidden, n_layers, attn_heads, dropout=0.1):
        super().__init__()
        self.num_features = num_features
        self.hidden = hidden
        self.n_layers = n_layers
        self.attn_heads = attn_heads
        self.dropout = dropout

        feed_forward_hidden = hidden * 4
        self.embedding = BERTEmbedding(self.num_features, dropout=dropout)

        encoder_layer = TransformerEncoderLayer(
            hidden,
            attn_heads,
            feed_forward_hidden,
            dropout
        )
        encoder_norm = LayerNorm(hidden)
        self.transformer_encoder = TransformerEncoder(
            encoder_layer,
            n_layers,
            encoder_norm,
            enable_nested_tensor=False
        )

        self.norm = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(hidden, hidden),
            nn.Dropout(0.1)
        )

    def forward(self, x, doy, mask, use_transformer=True):
        mask = mask == 0

        # Embedding
        x = self.embedding(input_sequence=x, doy_sequence=doy)

        if not use_transformer:
            # Saída só do embedding (para RandomForest)
            return x  

        # Com Transformer Encoder
        x = x.transpose(0, 1)  # [seq_length, batch_size, hidden]
        x = self.transformer_encoder(x, src_key_padding_mask=mask)
        x = x.transpose(0, 1)  # [batch_size, seq_length, hidden]

        return self.norm(x)
