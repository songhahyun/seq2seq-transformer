import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding module for transformer token embeddings."""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        """Precompute sinusoidal positional encodings for token embeddings."""
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
        """Add positional encodings to embedded sequences and apply dropout."""
        # x: [B, T, D]
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class Seq2SeqTransformer(nn.Module):
    """Encoder-decoder transformer for sequence-to-sequence translation."""

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
        """Initialize embeddings, positional encoders, transformer, and output head."""
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
        """Create a causal target mask that blocks attention to future tokens."""
        mask = torch.triu(torch.ones(size, size, device=device), diagonal=1).bool()
        return mask

    def make_src_key_padding_mask(self, src):
        """Build a source padding mask from PAD token positions."""
        # src: [B, T]
        return (src == self.pad_id)

    def make_tgt_key_padding_mask(self, tgt):
        """Build a target padding mask from PAD token positions."""
        return (tgt == self.pad_id)

    def forward(self, src, tgt_input):
        """Run teacher-forced encoder-decoder inference and return token logits."""
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
        """Decode each sequence by repeatedly selecting the highest-probability token."""
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

    @torch.no_grad()
    def beam_decode(
        self,
        src,
        bos_id: int,
        eos_id: int,
        max_len: int,
        beam_size: int = 5,
    ):
        """Decode each source sequence with beam search and pad results as a batch."""
        if beam_size < 1:
            raise ValueError("beam_size must be greater than or equal to 1")
        if beam_size == 1:
            return self.greedy_decode(src, bos_id, eos_id, max_len)

        self.eval()
        decoded = [
            self._beam_decode_single(
                src=src[i : i + 1],
                bos_id=bos_id,
                eos_id=eos_id,
                max_len=max_len,
                beam_size=beam_size,
            )
            for i in range(src.size(0))
        ]

        max_decoded_len = max(seq.size(0) for seq in decoded)
        padded = torch.full(
            (len(decoded), max_decoded_len),
            self.pad_id,
            dtype=torch.long,
            device=src.device,
        )
        for i, seq in enumerate(decoded):
            padded[i, : seq.size(0)] = seq
        return padded

    def _beam_decode_single(
        self,
        src,
        bos_id: int,
        eos_id: int,
        max_len: int,
        beam_size: int,
    ):
        """Run beam search for one source sequence and return the best token path."""
        device = src.device
        beams = torch.full((1, 1), bos_id, dtype=torch.long, device=device)
        beam_scores = torch.zeros(1, dtype=torch.float32, device=device)

        for _ in range(max_len - 1):
            finished = (beams == eos_id).any(dim=1)
            if finished.all():
                break

            src_beams = src.expand(beams.size(0), -1)
            logits = self.forward(src_beams, beams)
            log_probs = torch.log_softmax(logits[:, -1, :], dim=-1)

            candidate_sequences = []
            candidate_scores = []
            for beam_idx in range(beams.size(0)):
                if finished[beam_idx]:
                    next_seq = torch.cat(
                        [
                            beams[beam_idx],
                            torch.tensor([self.pad_id], dtype=torch.long, device=device),
                        ]
                    )
                    candidate_sequences.append(next_seq)
                    candidate_scores.append(beam_scores[beam_idx])
                    continue

                top_scores, top_tokens = torch.topk(log_probs[beam_idx], beam_size)
                for token_score, token_id in zip(top_scores, top_tokens):
                    next_seq = torch.cat([beams[beam_idx], token_id.view(1)])
                    candidate_sequences.append(next_seq)
                    candidate_scores.append(beam_scores[beam_idx] + token_score)

            candidate_scores = torch.stack(candidate_scores)
            candidate_sequences = torch.stack(candidate_sequences)
            best_scores, best_indices = torch.topk(
                candidate_scores,
                k=min(beam_size, candidate_scores.size(0)),
            )
            beams = candidate_sequences[best_indices]
            beam_scores = best_scores

        return beams[0]

    @torch.no_grad()
    def decode(
        self,
        src,
        bos_id: int,
        eos_id: int,
        max_len: int,
        strategy: str = "greedy",
        beam_size: int = 5,
    ):
        """Dispatch decoding to greedy search or beam search by strategy name."""
        if strategy == "greedy":
            return self.greedy_decode(src, bos_id, eos_id, max_len)
        if strategy == "beam":
            return self.beam_decode(src, bos_id, eos_id, max_len, beam_size=beam_size)
        raise ValueError(f"unsupported decode strategy: {strategy}")
