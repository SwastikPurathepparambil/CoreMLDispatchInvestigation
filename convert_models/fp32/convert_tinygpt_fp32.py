import math

import coremltools as ct
import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    def __init__(self, embed_dim=128, num_heads=2, seq_len=64):
        super().__init__()

        assert embed_dim % num_heads == 0

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.seq_len = seq_len

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        causal_mask = torch.tril(torch.ones(seq_len, seq_len))
        causal_mask = causal_mask.view(1, 1, seq_len, seq_len)
        self.register_buffer("causal_mask", causal_mask)

    def forward(self, x):
        batch_size = 1
        seq_len = self.seq_len
        embed_dim = self.embed_dim

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1))
        scores = scores / math.sqrt(self.head_dim)

        scores = scores.masked_fill(self.causal_mask == 0, -1e4)

        attn = F.softmax(scores, dim=-1)
        context = torch.matmul(attn, v)

        context = context.transpose(1, 2)
        context = context.reshape(batch_size, seq_len, embed_dim)

        return self.out_proj(context)


class TinyGPTBlock(nn.Module):
    def __init__(self, embed_dim=128, num_heads=2, seq_len=64):
        super().__init__()

        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = CausalSelfAttention(embed_dim, num_heads, seq_len)
        self.ln2 = nn.LayerNorm(embed_dim)

        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.GELU(),
            nn.Linear(4 * embed_dim, embed_dim),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class TinyGPTStyleModel(nn.Module):
    def __init__(
        self,
        vocab_size=4096,
        seq_len=64,
        embed_dim=128,
        num_heads=2,
        num_layers=2,
    ):
        super().__init__()

        self.seq_len = seq_len
        self.embed_dim = embed_dim

        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(seq_len, embed_dim)

        position_ids = torch.arange(seq_len, dtype=torch.int32).unsqueeze(0)
        self.register_buffer("position_ids", position_ids)

        self.blocks = nn.Sequential(
            *[
                TinyGPTBlock(embed_dim, num_heads, seq_len)
                for _ in range(num_layers)
            ]
        )

        self.final_ln = nn.LayerNorm(embed_dim)

    def forward(self, input_ids):
        token_embeds = self.token_embedding(input_ids)
        position_embeds = self.position_embedding(self.position_ids)

        x = token_embeds + position_embeds
        x = self.blocks(x)
        x = self.final_ln(x)

        return x


def main():
    model = TinyGPTStyleModel()
    model.eval()

    example_input = torch.randint(
        low=0,
        high=4096,
        size=(1, 64),
        dtype=torch.int32,
    )

    traced_model = torch.jit.trace(model, example_input, strict=False)

    mlmodel = ct.convert(
        traced_model,
        inputs=[
            ct.TensorType(
                name="input_ids",
                shape=example_input.shape,
                dtype=int,
            )
        ],
        convert_to="mlprogram",
        compute_precision=ct.precision.FLOAT32,
        minimum_deployment_target=ct.target.macOS13,
    )

    output_path = "models/TinyGPT_fp32.mlpackage"
    mlmodel.save(output_path)

    print(f"Saved Core ML model to {output_path}")


if __name__ == "__main__":
    main()