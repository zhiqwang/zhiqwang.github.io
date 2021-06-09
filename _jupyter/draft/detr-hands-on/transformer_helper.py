import io
import math
from collections import Counter

import torch
from torch import nn, Tensor
from torch.nn import (
    TransformerEncoder,
    TransformerDecoder,
    TransformerEncoderLayer,
    TransformerDecoderLayer,
)
from torch.nn.utils.rnn import pad_sequence
from torchtext.vocab import Vocab


class Seq2SeqTransformer(nn.Module):
    def __init__(
        self,
        num_encoder_layers: int,
        num_decoder_layers: int,
        num_head: int,
        emb_size: int,
        src_vocab_size: int,
        tgt_vocab_size: int,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.src_tok_emb = TokenEmbedding(src_vocab_size, emb_size)
        self.tgt_tok_emb = TokenEmbedding(tgt_vocab_size, emb_size)
        self.positional_encoding = PositionalEncoding(emb_size, dropout=dropout)

        encoder_layer = TransformerEncoderLayer(
            d_model=emb_size,
            nhead=num_head,
            dim_feedforward=dim_feedforward,
        )
        self.transformer_encoder = TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers,
        )
        decoder_layer = TransformerDecoderLayer(
            d_model=emb_size,
            nhead=num_head,
            dim_feedforward=dim_feedforward,
        )
        self.transformer_decoder = TransformerDecoder(
            decoder_layer,
            num_layers=num_decoder_layers,
        )

        self.generator = nn.Linear(emb_size, tgt_vocab_size)

    def forward(
        self,
        src: Tensor,
        tgt: Tensor,
        src_mask: Tensor,
        tgt_mask: Tensor,
        src_padding_mask: Tensor,
        tgt_padding_mask: Tensor,
        memory_key_padding_mask: Tensor,
    ):
        src_emb = self.positional_encoding(self.src_tok_emb(src))
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(tgt))
        memory = self.transformer_encoder(src_emb, src_mask, src_padding_mask)
        outs = self.transformer_decoder(
            tgt_emb, memory, tgt_mask, None, tgt_padding_mask, memory_key_padding_mask)
        return self.generator(outs)

    def encode(self, src: Tensor, src_mask: Tensor):
        src_emb = self.positional_encoding(self.src_tok_emb(src))
        return self.transformer_encoder(src_emb, src_mask)

    def decode(self, tgt: Tensor, memory: Tensor, tgt_mask: Tensor):
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(tgt))
        return self.transformer_decoder(tgt_emb, memory, tgt_mask)


class PositionalEncoding(nn.Module):
    """
    Text tokens are represented by using token embeddings. Positional encoding
    is added to the token embedding to introduce a notion of word order.
    """
    def __init__(self, emb_size: int, dropout, maxlen: int = 5000):
        super().__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2) * math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: Tensor):
        return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, emb_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size
    def forward(self, tokens: Tensor):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)


def build_vocab(filepath, tokenizer):
    counter = Counter()
    with io.open(filepath, encoding="utf8") as f:
        for string_ in f:
            counter.update(tokenizer(string_))
    return Vocab(counter, specials=['<unk>', '<pad>', '<bos>', '<eos>'])


def data_process(filepaths, src_vocab, src_tokenizer, tgt_vocab, tgt_tokenizer):
    raw_fr_iter = iter(io.open(filepaths[0], encoding="utf8"))
    raw_en_iter = iter(io.open(filepaths[1], encoding="utf8"))
    data = []
    for (raw_fr, raw_en) in zip(raw_fr_iter, raw_en_iter):
        src_tensor = torch.tensor(
            [src_vocab[token] for token in src_tokenizer(raw_fr.rstrip("\n"))],
            dtype=torch.long,
        )
        tgt_tensor = torch.tensor(
            [tgt_vocab[token] for token in tgt_tokenizer(raw_en.rstrip("\n"))],
            dtype=torch.long,
        )
        data.append((src_tensor, tgt_tensor))
    return data


def generate_batch(data_batch, start_symbol, end_symbol, padding_symbol):
    fr_batch, en_batch = [], []
    for (fr_item, en_item) in data_batch:
        fr_batch.append(
            torch.cat([torch.tensor([start_symbol]), fr_item, torch.tensor([end_symbol])], dim=0))
        en_batch.append(
            torch.cat([torch.tensor([start_symbol]), en_item, torch.tensor([end_symbol])], dim=0))

    fr_batch = pad_sequence(fr_batch, padding_value=padding_symbol)
    en_batch = pad_sequence(en_batch, padding_value=padding_symbol)
    return fr_batch, en_batch


def generate_square_subsequent_mask(sz, device):
    """
    Create a ``subsequent word`` mask to stop a target word from attending to its subsequent words.
    """
    mask = (torch.triu(torch.ones((sz, sz), device=device)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


def create_mask(src, tgt, padding_symbol, device):
    """
    Create masks, for masking source and target padding tokens.
    """
    src_seq_len = src.shape[0]
    tgt_seq_len = tgt.shape[0]

    tgt_mask = generate_square_subsequent_mask(tgt_seq_len, device)
    src_mask = torch.zeros((src_seq_len, src_seq_len), device=device).type(torch.bool)

    src_padding_mask = (src == padding_symbol).transpose(0, 1)
    tgt_padding_mask = (tgt == padding_symbol).transpose(0, 1)
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask


def loss_func(padding_symbol):
    return torch.nn.CrossEntropyLoss(ignore_index=padding_symbol)


def train_epoch(model, train_iter, optimizer, padding_symbol, device):
    model.train()
    losses = 0
    for (src, tgt) in train_iter:
        src = src.to(device)
        tgt = tgt.to(device)

        tgt_input = tgt[:-1, :]

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(
            src, tgt_input, padding_symbol, device)

        logits = model(src, tgt_input, src_mask, tgt_mask,
                       src_padding_mask, tgt_padding_mask, src_padding_mask)

        optimizer.zero_grad()

        tgt_out = tgt[1:, :]
        loss_fn = loss_func(padding_symbol)
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        loss.backward()

        optimizer.step()
        losses += loss.item()
    return losses / len(train_iter)


@torch.no_grad()
def evaluate(model, val_iter, padding_symbol, device):
    model.eval()
    losses = 0
    for idx, (src, tgt) in enumerate(val_iter):
        src = src.to(device)
        tgt = tgt.to(device)

        tgt_input = tgt[:-1, :]

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(
            src, tgt_input, padding_symbol, device)

        logits = model(src, tgt_input, src_mask, tgt_mask,
                       src_padding_mask, tgt_padding_mask, src_padding_mask)
        tgt_out = tgt[1:,:]
        loss_fn = loss_func(padding_symbol)
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        losses += loss.item()
    return losses / len(val_iter)


def greedy_decode(model, src, src_mask, max_len, start_symbol, end_symbol, device):
    src = src.to(device)
    src_mask = src_mask.to(device)

    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(device)
    for i in range(max_len - 1):
        memory = memory.to(device)
        # memory_mask = torch.zeros(ys.shape[0], memory.shape[0]).to(device).type(torch.bool)
        tgt_mask = generate_square_subsequent_mask(ys.size(0), device).type(torch.bool)
        out = model.decode(ys, memory, tgt_mask)
        out = out.transpose(0, 1)
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.item()

        ys = torch.cat([ys, torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=0)
        if next_word == end_symbol:
            break
    return ys


@torch.no_grad()
def translate(model, src, src_vocab, tgt_vocab, src_tokenizer, start_symbol, end_symbol, device):
    model.eval()
    tokens = [start_symbol] + [src_vocab.stoi[tok] for tok in src_tokenizer(src)] + [end_symbol]
    num_tokens = len(tokens)
    src = (torch.LongTensor(tokens).reshape(num_tokens, 1))
    src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool)
    tgt_tokens = greedy_decode(model, src, src_mask, max_len=num_tokens + 5,
                               start_symbol=start_symbol, end_symbol=end_symbol,
                               device=device).flatten()
    tgt_language = " ".join([tgt_vocab.itos[tok] for tok in tgt_tokens])
    tgt_language = tgt_language.replace("<bos> ", "").replace(" <eos>", "")
    return tgt_language
