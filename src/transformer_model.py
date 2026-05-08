import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)  # [T, D]
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)  # [T, 1]
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # [1, T, D]
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x: [B, T, D]
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class Seq2SeqTransformer(nn.Module):
    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int,
        nhead: int,
        num_encoder_layers: int,
        num_decoder_layers: int,
        dim_feedforward: int,
        dropout: float,
        pad_id: int,
    ):
        super().__init__()
        self.d_model = d_model
        self.pad_id = pad_id

        self.src_embedding = nn.Embedding(src_vocab_size, d_model, padding_idx=pad_id)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model, padding_idx=pad_id)

        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)
        self.pos_decoder = PositionalEncoding(d_model, dropout=dropout)

        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )

        self.output_proj = nn.Linear(d_model, tgt_vocab_size)

    def generate_square_subsequent_mask(self, size, device):
        mask = torch.triu(torch.ones(size, size, device=device), diagonal=1).bool()
        return mask

    def make_src_key_padding_mask(self, src):
        # src: [B, T]
        return (src == self.pad_id)

    def make_tgt_key_padding_mask(self, tgt):
        return (tgt == self.pad_id)

    def forward(self, src, tgt_input):
        # src: [B, S]
        # tgt_input: [B, T]
        device = src.device

        src_key_padding_mask = self.make_src_key_padding_mask(src)
        tgt_key_padding_mask = self.make_tgt_key_padding_mask(tgt_input)
        tgt_mask = self.generate_square_subsequent_mask(tgt_input.size(1), device)

        src_emb = self.src_embedding(src) * math.sqrt(self.d_model)
        tgt_emb = self.tgt_embedding(tgt_input) * math.sqrt(self.d_model)

        src_emb = self.pos_encoder(src_emb)
        tgt_emb = self.pos_decoder(tgt_emb)

        output = self.transformer(
            src=src_emb,
            tgt=tgt_emb,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask,
        )

        logits = self.output_proj(output)  # [B, T, vocab]
        return logits

    @torch.no_grad()
    def greedy_decode(self, src, bos_id: int, eos_id: int, max_len: int):
        self.eval()
        device = src.device
        batch_size = src.size(0)

        generated = torch.full(
            (batch_size, 1),
            bos_id,
            dtype=torch.long,
            device=device,
        )

        for _ in range(max_len - 1):
            logits = self.forward(src, generated)  # [B, T, vocab]
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)  # [B, 1]
            generated = torch.cat([generated, next_token], dim=1)

            if (next_token == eos_id).all():
                break

        return generated