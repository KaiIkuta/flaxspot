import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.training import train_state
import optax
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

class PositionalEncoding(nn.Module):
    max_len: int = 5000
    d_model: int = 64

    def setup(self):
        pe = jnp.zeros((self.max_len, self.d_model))
        position = jnp.arange(0, self.max_len, dtype=jnp.float32)[:, jnp.newaxis]
        div_term = jnp.exp(jnp.arange(0, self.d_model, 2, dtype=jnp.float32) * -(jnp.log(10000.0) / self.d_model))
        pe = pe.at[:, 0::2].set(jnp.sin(position * div_term))
        pe = pe.at[:, 1::2].set(jnp.cos(position * div_term))
        self.pe = pe[jnp.newaxis, ...]

    def __call__(self, x, start_idx=0):
        seq_len = x.shape[1]
        return x + self.pe[:, start_idx : start_idx + seq_len, :]

class TransformerEncoderBlock(nn.Module):
    num_heads: int = 4
    qkv_features: int = 64
    mlp_dim: int = 128
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, inputs, attn_mask=None, deterministic=True):
        x = nn.LayerNorm()(inputs)
        attn_out = nn.SelfAttention(
            num_heads=self.num_heads, qkv_features=self.qkv_features,
            deterministic=deterministic)(x, mask=attn_mask)
        x = inputs + nn.Dropout(self.dropout_rate)(attn_out, deterministic=deterministic)
        
        y = nn.LayerNorm()(x)
        y = nn.Dense(self.mlp_dim)(y)
        y = nn.gelu(y)
        y = nn.Dense(inputs.shape[-1])(y)
        return x + nn.Dropout(self.dropout_rate)(y, deterministic=deterministic)

class TransformerDecoderBlock(nn.Module):
    num_heads: int = 4
    qkv_features: int = 64
    mlp_dim: int = 128
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, inputs, memory, memory_mask=None, deterministic=True):
        # Self Attention
        x = nn.LayerNorm()(inputs)
        attn_out = nn.SelfAttention(
            num_heads=self.num_heads, qkv_features=self.qkv_features,
            deterministic=deterministic)(x)
        x = inputs + nn.Dropout(self.dropout_rate)(attn_out, deterministic=deterministic)
        
        # Cross Attention
        y = nn.LayerNorm()(x)
        cross_out = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads, qkv_features=self.qkv_features,
            deterministic=deterministic)(y, memory, mask=memory_mask)
        x = x + nn.Dropout(self.dropout_rate)(cross_out, deterministic=deterministic)

        # FFN
        y = nn.LayerNorm()(x)
        y = nn.Dense(self.mlp_dim)(y)
        y = nn.gelu(y)
        y = nn.Dense(inputs.shape[-1])(y)
        return x + nn.Dropout(self.dropout_rate)(y, deterministic=deterministic)


class SOTATimeSeriesVAE(nn.Module):
    seq_len: int
    d_model: int = 64
    latent_dim: int = 32
    num_layers: int = 3
    dropout_rate: float = 0.1

    def setup(self):
        self.input_proj = nn.Dense(self.d_model)
        self.pos_encoder = PositionalEncoding(d_model=self.d_model)
        self.cls_token = self.param('cls_token', jax.nn.initializers.normal(stddev=0.02), (1, 1, self.d_model))
        
        self.encoder_blocks = [TransformerEncoderBlock(dropout_rate=self.dropout_rate) for _ in range(self.num_layers)]
        self.fc_mu = nn.Dense(self.latent_dim)
        self.fc_logvar = nn.Dense(self.latent_dim)
        
        self.memory_num_tokens = 4
        self.latent_to_memory = nn.Dense(self.memory_num_tokens * self.d_model)
        self.query_embed = self.param('query_embed', jax.nn.initializers.normal(stddev=0.02), (1, self.seq_len, self.d_model))
        
        self.decoder_blocks = [TransformerDecoderBlock(dropout_rate=self.dropout_rate) for _ in range(self.num_layers)]
        self.out_mu = nn.Dense(1)
        self.out_logvar = nn.Dense(1)

    def encode(self, x, padding_mask, deterministic=True):
        batch_size = x.shape[0]
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        
        cls_tokens = jnp.broadcast_to(self.cls_token, (batch_size, 1, self.d_model))
        x = jnp.concatenate([cls_tokens, x], axis=1)
        
        cls_mask = jnp.ones((batch_size, 1), dtype=padding_mask.dtype)
        full_mask = jnp.concatenate([cls_mask, padding_mask], axis=1)
        attn_mask = nn.make_attention_mask(full_mask > 0, full_mask > 0)
        
        for block in self.encoder_blocks:
            x = block(x, attn_mask=attn_mask, deterministic=deterministic)
        
        cls_out = x[:, 0, :]
        mu = self.fc_mu(cls_out)
        logvar = self.fc_logvar(cls_out)
        return mu, logvar

    def reparameterize(self, rng, mu, logvar):
        std = jnp.exp(0.5 * logvar)
        eps = jax.random.normal(rng, logvar.shape)
        return mu + eps * std

    def decode(self, z, deterministic=True):
        batch_size = z.shape[0]
        memory = self.latent_to_memory(z).reshape((batch_size, self.memory_num_tokens, self.d_model))
        
        queries = jnp.broadcast_to(self.query_embed, (batch_size, self.seq_len, self.d_model))
        queries = self.pos_encoder(queries)
        
        for block in self.decoder_blocks:
            queries = block(queries, memory, memory_mask=None, deterministic=deterministic)
        
        pred_mu = self.out_mu(queries)
        pred_logvar = self.out_logvar(queries)
        return pred_mu, pred_logvar

    def __call__(self, x, padding_mask, rng, deterministic=True):
        mu, logvar = self.encode(x, padding_mask, deterministic)
        z = self.reparameterize(rng, mu, logvar)
        pred_mu, pred_logvar = self.decode(z, deterministic)
        return (pred_mu, pred_logvar), mu, logvar


def sota_loss_fn(params, model, inputs, padding_mask, rng, beta=1.0, free_bits=0.5):
    (pred_mu, pred_logvar), mu, logvar = model.apply(
        {'params': params}, inputs, padding_mask, rng, 
        deterministic=False, rngs={'dropout': rng}
    )
    mask_expanded = jnp.expand_dims(padding_mask, axis=-1)
    

    nll = 0.5 * (jnp.exp(-pred_logvar) * ((inputs - pred_mu) ** 2) + pred_logvar)
    sum_nll = jnp.sum(nll * mask_expanded, axis=(1, 2))
    count_valid = jnp.clip(jnp.sum(mask_expanded, axis=(1, 2)), a_min=1e-9)
    recon_loss = jnp.mean(sum_nll / count_valid)
    

    kl_per_dim = -0.5 * (1 + logvar - jnp.square(mu) - jnp.exp(logvar))
    kl_loss = jnp.mean(jnp.sum(jnp.maximum(kl_per_dim, free_bits), axis=1))
    
    return recon_loss + beta * kl_loss, (recon_loss, kl_loss)


def plot_latent_space(latents, labels=None):
    perplexity = min(30, latents.shape[0] - 1)
    reducer = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    latents_2d = reducer.fit_transform(latents)

    plt.figure(figsize=(8, 6))
    if labels is not None:
        scatter = plt.scatter(latents_2d[:, 0], latents_2d[:, 1], c=labels, cmap='viridis', alpha=0.8, s=50)
        plt.colorbar(scatter, label="Sample Index (Dummy Feature)")
    else:
        plt.scatter(latents_2d[:, 0], latents_2d[:, 1], alpha=0.8, s=50)

    plt.title("t-SNE Latent Space Map", fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

def plot_latent_traversal(model, params, base_latent, target_dim, seq_len, traversal_range=(-3.0, 3.0), num_steps=5):
    z_values = np.linspace(traversal_range[0], traversal_range[1], num_steps)
    traversal_z = np.tile(base_latent, (num_steps, 1))
    traversal_z[:, target_dim] = z_values
    
    jax_z = jnp.array(traversal_z)
    
    pred_mu, _ = jax.jit(lambda z: model.apply({'params': params}, z, deterministic=True, method=model.decode))(jax_z)
    
    reconstructed_flux = np.array(pred_mu).squeeze(-1)

    fig, axes = plt.subplots(num_steps, 1, figsize=(10, 1.2 * num_steps), sharex=True, sharey=True)
    for i in range(num_steps):
        axes[i].plot(reconstructed_flux[i], color='crimson', linewidth=1.5)
        axes[i].set_title(f"z_{target_dim} = {z_values[i]:.2f}", fontsize=10, loc='right')
        axes[i].grid(True, linestyle='--', alpha=0.3)
        axes[i].set_ylabel("Flux")

    axes[-1].set_xlabel("Time Steps", fontsize=12)
    plt.suptitle(f"Latent Traversal of Dimension {target_dim}", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    SEQ_LEN = 256
    BATCH_SIZE = 64
    LATENT_DIM = 32
    
    # 準備
    rng = jax.random.PRNGKey(42)
    model = SOTATimeSeriesVAE(seq_len=SEQ_LEN, latent_dim=LATENT_DIM)
    

    np.random.seed(42)
    dummy_flux = np.random.randn(BATCH_SIZE, SEQ_LEN, 1).astype(np.float32)
    dummy_mask = np.random.choice([0.0, 1.0], size=(BATCH_SIZE, SEQ_LEN), p=[0.2, 0.8]).astype(np.float32)
    dummy_flux = dummy_flux * dummy_mask[:, :, np.newaxis]

    jax_flux = jnp.array(dummy_flux)
    jax_mask = jnp.array(dummy_mask)


    rng, init_rng = jax.random.split(rng)
    init_params = model.init({'params': init_rng}, jax_flux[:1], jax_mask[:1], init_rng, deterministic=True)['params']
    tx = optax.adamw(learning_rate=1e-3, weight_decay=1e-4)
    state = train_state.TrainState.create(apply_fn=model.apply, params=init_params, tx=tx)


    @jax.jit
    def train_step(state, inputs, padding_mask, rng, beta):
        rng, step_rng = jax.random.split(rng)
        def compute_loss(params):
            return sota_loss_fn(params, model, inputs, padding_mask, step_rng, beta=beta)
        (loss, aux), grads = jax.value_and_grad(compute_loss, has_aux=True)(state.params)
        return state.apply_gradients(grads=grads), loss, aux[0], aux[1], rng


    print("--- Training Started ---")
    for epoch in range(1, 11):
        current_beta = jnp.array(min(1.0, epoch / 5.0))
        state, loss, r_loss, k_loss, rng = train_step(state, jax_flux, jax_mask, rng, current_beta)
        if epoch % 2 == 0 or epoch == 1:
            print(f"Epoch {epoch:02d} | Loss: {loss:.4f} | Recon: {r_loss:.4f} | KL: {k_loss:.4f} | Beta: {current_beta:.1f}")

    @jax.jit
    def extract_features(params, inputs, padding_mask):
        mu, _ = model.apply({'params': params}, inputs, padding_mask, deterministic=True, method=model.encode)
        return mu

    latents = extract_features(state.params, jax_flux, jax_mask)
    np_latents = np.array(latents)
    print(f"\nExtracted Latents Shape: {np_latents.shape}")

    print("Generating Latent Space Map...")
    plot_latent_space(np_latents, labels=np.arange(BATCH_SIZE))

    print("Generating Latent Traversal Plot...")
    plot_latent_traversal(
        model=model, params=state.params, 
        base_latent=np_latents[0], target_dim=0, 
        seq_len=SEQ_LEN
    )
