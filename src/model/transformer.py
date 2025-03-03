import math
import torch
from torch import nn, Tensor
from torch.nn import Embedding, Transformer

from src.model.positional_encodings import PositionalEncoding
from src.dataset import CustomDataset

class TranslateTransformer(nn.Module):
    def __init__(
        self,
        n_vocab_source: int,
        n_vocab_dest: int,
        pad_idx: int,
        d_vocab: int = None,
        d_model: int = 512,
        nhead: int = 8,
        num_encoder_layers: int = 3,
        num_decoder_layers: int = 3,
        dim_feedforward: int = 1024,
        dropout: float = 0.3,
        weight_sharing: bool = False,
    ):
        super(TranslateTransformer, self).__init__()
        if d_vocab is None:
            d_vocab = d_model
        self.d_vocab = d_vocab
        self.weight_sharing = weight_sharing

        self.source_embeddings = Embedding(num_embeddings=n_vocab_source, embedding_dim=d_vocab, padding_idx=pad_idx)
        self.dest_embeddings = Embedding(num_embeddings=n_vocab_dest, embedding_dim=d_vocab, padding_idx=pad_idx)
        self.positional_encodings = PositionalEncoding(input_dim=d_vocab, dropout_rate=dropout)
        self.transformer = Transformer(d_model=d_model,
                                       nhead=nhead,
                                       num_encoder_layers=num_encoder_layers,
                                       num_decoder_layers=num_decoder_layers,
                                       dim_feedforward=dim_feedforward,
                                       dropout=dropout,
                                       batch_first=True)
        self.fc = nn.Linear(d_model, n_vocab_dest)

        self.pad_idx = pad_idx
        
        self.init()

        if self.weight_sharing:
            self.fc.weight = self.dest_embeddings.weight
            # self.source_embeddings.weight = self.dest_embeddings.weight

    def init(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        
        nn.init.normal_(self.source_embeddings.weight, 0, 1 / math.sqrt(self.d_vocab))
        nn.init.normal_(self.dest_embeddings.weight, 0, 1 / math.sqrt(self.d_vocab))
        nn.init.normal_(self.fc.weight, 0, 1 / math.sqrt(self.d_vocab))
    
    def forward(self, source: Tensor, dest: Tensor, **batch) -> Tensor:
        source_embed = self.positional_encodings(self.source_embeddings(source) * math.sqrt(self.d_vocab))
        dest_embed = self.positional_encodings(self.dest_embeddings(dest) * math.sqrt(self.d_vocab))

        output = self.transformer(
            src=source_embed,
            tgt=dest_embed,
            tgt_mask=self._generate_attn_mask(dest),
            src_key_padding_mask=self._generate_padding_mask(source),
            tgt_key_padding_mask=self._generate_padding_mask(dest),
        )

        return {"output": self.fc(output)}

    def encode(self, source: Tensor) -> Tensor:
        source_embed = self.positional_encodings(self.source_embeddings(source) * math.sqrt(self.d_vocab))

        return self.transformer.encoder(source_embed)
    
    def decode(self, dest: Tensor, memory: Tensor) -> Tensor:
        dest_embed = self.positional_encodings(self.dest_embeddings(dest) * math.sqrt(self.d_vocab))

        output = self.transformer.decoder(
            tgt=dest_embed,
            memory=memory,
        )

        return self.fc(output[:, -1, :])
    
    def translate(self, source: Tensor, length: int, max_length: int = 200, **batch):
        source = source[:length]
        if source.ndim == 1:
            source = source.unsqueeze(0)

        memory = self.encode(source)
        output = torch.LongTensor([[CustomDataset.BOS]]).to(source.device)

        for _ in range(1, max_length):
            logits = self.decode(output, memory)
            token = logits.argmax(dim=-1).item()            

            output = torch.cat([output, torch.LongTensor([[token]]).to(source.device)], dim=-1)

            if token == CustomDataset.EOS:
                break

        return output

    def _generate_padding_mask(self, x: Tensor) -> Tensor:
        return x == self.pad_idx

    def _generate_attn_mask(self, x: Tensor) -> Tensor:
        size = x.shape[1]

        return torch.triu(torch.full((size, size), float('-inf')), 1).to(x.device)
