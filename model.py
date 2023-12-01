import torch
import torch.nn as nn

from torch.nn import functional as F


class SelfAttentionHead(nn.Module):
    def __init__(self, head_size: int, block_size: int, num_embeddings: int, dropout: float):
        super().__init__()
        self.key = nn.Linear(num_embeddings, head_size, bias=False)
        self.query = nn.Linear(num_embeddings, head_size, bias=False)
        self.value = nn.Linear(num_embeddings, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        b, t, c = x.shape
        key: nn.Linear = self.key(x)  # (B,T,hs)
        query: nn.Linear = self.query(x)  # (B,T,hs)

        # compute attention scores ("affinities")
        weights: torch.Tensor = query @ key.transpose(-2, -1) * key.shape[-1] ** -0.5  # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        weights: torch.Tensor = weights.masked_fill(self.tril[:t, :t] == 0, float('-inf'))  # (B, T, T)
        weights: torch.Tensor = F.softmax(weights, dim=-1)  # (B, T, T)
        weights: torch.Tensor = self.dropout(weights)

        # perform the weighted aggregation of the values
        values: torch.Tensor = self.value(x)  # (B,T,hs)

        output: torch.Tensor = weights @ values  # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return output


class MultiAttentionHead(nn.Module):
    def __init__(self, head_size: int, num_heads: int, num_embeddings: int, dropout: float, block_size: int):
        super().__init__()
        self.heads = nn.ModuleList(
            [SelfAttentionHead(head_size=head_size, num_embeddings=num_embeddings, block_size=block_size, dropout=dropout) for _ in range(num_heads)])
        self.projection = nn.Linear(head_size * num_heads, num_embeddings)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        output: torch.Tensor = torch.cat([head(x) for head in self.heads], dim=-1)
        output: torch.Tensor = self.dropout(self.projection(output))

        return output


class FeedForwardNetwork(nn.Module):
    def __init__(self, num_embeddings: int, dropout: float):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(num_embeddings, 4 * num_embeddings),
            nn.ReLU(),
            nn.Linear(4 * num_embeddings, num_embeddings),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.network(x)


class TransformerBlock(nn.Module):
    def __init__(self, num_embeddings: int, num_head: int, dropout: float, block_size: int):
        super().__init__()

        head_size: int = num_embeddings // num_head
        self.self_attention = MultiAttentionHead(head_size=head_size, num_heads=num_head, num_embeddings=num_embeddings, dropout=dropout,
                                                 block_size=block_size)
        self.feed_forward = FeedForwardNetwork(num_embeddings=num_embeddings, dropout=dropout)
        self.layer_norm_1 = nn.LayerNorm(num_embeddings)
        self.layer_norm_2 = nn.LayerNorm(num_embeddings)

    def forward(self, x):
        x = x + self.self_attention(self.layer_norm_1(x))
        x = x + self.feed_forward(self.layer_norm_2(x))

        return x


class NanoGPTModel(nn.Module):
    def __init__(self, vocab_size: int, num_embeddings: int, block_size: int, num_head: int, num_layer: int, dropout: float, device):
        super().__init__()

        self.device = device
        self.block_size = block_size
        self.token_embedding_table = nn.Embedding(vocab_size, num_embeddings)
        self.position_embedding_table = nn.Embedding(block_size, num_embeddings)
        self.blocks = nn.Sequential(*[TransformerBlock(num_embeddings, num_head, dropout, block_size=block_size) for _ in range(num_layer)])
        self.layer_norm_final = nn.LayerNorm(num_embeddings)
        self.language_model_head = nn.Linear(num_embeddings, vocab_size)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        b, t = idx.shape

        # idx and targets are both (B,T) tensor of integers
        token_embeddings = self.token_embedding_table(idx.to(self.device))  # (B,T,C)
        position_embeddings = self.position_embedding_table(torch.arange(t, device=self.device))  # (T,C)
        x = token_embeddings + position_embeddings  # (B,T,C)
        x = self.blocks(x)  # (B,T,C)
        x = self.layer_norm_final(x)  # (B,T,C)
        logits = self.language_model_head(x)  # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            b, t, c = logits.shape
            logits = logits.view(b * t, c).to(self.device)
            targets = targets.view(b * t).to(self.device)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -self.block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :]  # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx
