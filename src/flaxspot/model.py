import jax
import jax.numpy as jnp
from flax import linen as nn


class PositionalEncoding(nn.Module):
    max_len: int = 50000
    d_model: int = 128

    def setup(self):
        pe = jnp.zeros((self.max_len, self.d_model))
        position = jnp.arange(0, self.max_len, dtype=jnp.float32)[:, jnp.newaxis]
        div_term = jnp.exp(
            jnp.arange(0, self.d_model, 2, dtype=jnp.float32) * -(jnp.log(10000.0) / self.d_model)
        )
        pe = pe.at[:, 0::2].set(jnp.sin(position * div_term))
        pe = pe.at[:, 1::2].set(jnp.cos(position * div_term))
        self.pe = pe[jnp.newaxis, ...]

    def __call__(self, x, start_idx=0):
        seq_len = x.shape[1]
        return x + self.pe[:, start_idx : start_idx + seq_len, :]


class AstroconformerEncoderBlock(nn.Module):
    num_heads: int = 4
    qkv_features: int = 128
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, inputs, attn_mask=None, deterministic=True):
        # 1. Multi-Head Self-Attention
        x = nn.LayerNorm()(inputs)
        attn_out = nn.SelfAttention(
            num_heads=self.num_heads,
            qkv_features=self.qkv_features,
            deterministic=deterministic,
        )(x, mask=attn_mask)
        x = inputs + nn.Dropout(self.dropout_rate)(attn_out, deterministic=deterministic)

        # 2. 1D Convolution 1
        y = nn.LayerNorm()(x)
        y = nn.Conv(features=inputs.shape[-1], kernel_size=(3,), padding="SAME")(y)
        y = nn.LayerNorm()(y)
        y = jax.nn.silu(y)
        x = x + nn.Dropout(self.dropout_rate)(y, deterministic=deterministic)

        # 3. 1D Convolution 2
        z = nn.LayerNorm()(x)
        z = nn.Conv(features=inputs.shape[-1], kernel_size=(3,), padding="SAME")(z)
        z = nn.LayerNorm()(z)
        z = jax.nn.silu(z)
        x = x + nn.Dropout(self.dropout_rate)(z, deterministic=deterministic)

        return x


class TransformerDecoderBlock(nn.Module):
    num_heads: int = 4
    qkv_features: int = 128
    mlp_dim: int = 256
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, inputs, memory, memory_mask=None, deterministic=True):
        # Self-Attention
        x = nn.LayerNorm()(inputs)
        attn_out = nn.SelfAttention(
            num_heads=self.num_heads,
            qkv_features=self.qkv_features,
            deterministic=deterministic,
        )(x)
        x = inputs + nn.Dropout(self.dropout_rate)(attn_out, deterministic=deterministic)

        # Cross-Attention
        y = nn.LayerNorm()(x)
        cross_out = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            qkv_features=self.qkv_features,
            deterministic=deterministic,
        )(y, memory, mask=memory_mask)
        x = x + nn.Dropout(self.dropout_rate)(cross_out, deterministic=deterministic)

        # Feed-Forward Network
        y = nn.LayerNorm()(x)
        y = nn.Dense(self.mlp_dim)(y)
        y = nn.gelu(y)
        y = nn.Dense(inputs.shape[-1])(y)
        return x + nn.Dropout(self.dropout_rate)(y, deterministic=deterministic)


class AstroconformerVAE(nn.Module):
    seq_len: int = 4000
    patch_size: int = 16
    d_model: int = 128
    latent_dim: int = 64
    memory_num_tokens: int = 128
    num_layers: int = 3
    num_heads: int = 4
    dropout_rate: float = 0.1

    def setup(self):
        self.patch_embed = nn.Dense(self.d_model)
        self.pos_encoder = PositionalEncoding(d_model=self.d_model)
        self.cls_token = self.param(
            "cls_token", jax.nn.initializers.normal(stddev=0.02), (1, 1, self.d_model)
        )

        self.encoder_blocks = [
            AstroconformerEncoderBlock(
                num_heads=self.num_heads,
                qkv_features=self.d_model,
                dropout_rate=self.dropout_rate,
            )
            for _ in range(self.num_layers)
        ]
        self.fc_mu = nn.Dense(self.latent_dim)
        self.fc_logvar = nn.Dense(self.latent_dim)

        self.latent_to_memory = nn.Dense(self.memory_num_tokens * self.d_model)
        self.query_embed = self.param(
            "query_embed",
            jax.nn.initializers.normal(stddev=0.02),
            (1, self.seq_len, self.d_model),
        )

        self.decoder_blocks = [
            TransformerDecoderBlock(
                num_heads=self.num_heads,
                qkv_features=self.d_model,
                mlp_dim=self.d_model * 2,
                dropout_rate=self.dropout_rate,
            )
            for _ in range(self.num_layers)
        ]
        self.out_mu = nn.Dense(1)
        self.out_logvar = nn.Dense(1)

    def encode(self, x, padding_mask, deterministic=True):
        batch_size = x.shape[0]
        num_patches = self.seq_len // self.patch_size
        x_patched = x.reshape((batch_size, num_patches, self.patch_size * x.shape[-1]))
        x_emb = self.patch_embed(x_patched)
        x_emb = self.pos_encoder(x_emb)

        cls_tokens = jnp.broadcast_to(self.cls_token, (batch_size, 1, self.d_model))
        x_emb = jnp.concatenate([cls_tokens, x_emb], axis=1)

        mask_patched = padding_mask.reshape((batch_size, num_patches, self.patch_size))
        patch_mask = jnp.max(mask_patched, axis=-1)
        cls_mask = jnp.ones((batch_size, 1), dtype=patch_mask.dtype)
        full_mask = jnp.concatenate([cls_mask, patch_mask], axis=1)
        attn_mask = nn.make_attention_mask(full_mask > 0, full_mask > 0)

        for block in self.encoder_blocks:
            x_emb = block(x_emb, attn_mask=attn_mask, deterministic=deterministic)

        cls_out = x_emb[:, 0, :]
        mu = self.fc_mu(cls_out)
        logvar = self.fc_logvar(cls_out)
        return mu, logvar

    def reparameterize(self, rng, mu, logvar):
        std = jnp.exp(0.5 * logvar)
        eps = jax.random.normal(rng, logvar.shape)
        return mu + eps * std

    def decode(self, z, deterministic=True):
        batch_size = z.shape[0]
        memory = self.latent_to_memory(z).reshape(
            (batch_size, self.memory_num_tokens, self.d_model)
        )
        queries = jnp.broadcast_to(
            self.query_embed, (batch_size, self.seq_len, self.d_model)
        )
        queries = self.pos_encoder(queries)

        for block in self.decoder_blocks:
            queries = block(
                queries, memory, memory_mask=None, deterministic=deterministic
            )

        pred_mu = self.out_mu(queries)
        pred_logvar = self.out_logvar(queries)
        return pred_mu, pred_logvar

    def __call__(self, x, padding_mask, rng, deterministic=True):
        mu, logvar = self.encode(x, padding_mask, deterministic)
        z = self.reparameterize(rng, mu, logvar)
        pred_mu, pred_logvar = self.decode(z, deterministic)
        return (pred_mu, pred_logvar), mu, logvar
