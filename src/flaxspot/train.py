import numpy as np
import jax
import jax.numpy as jnp
from flax.training import train_state
import optax
from .model import AstroconformerVAE


def sota_loss_fn(params, model, inputs, padding_mask, rng, beta=1.0, free_bits=0.5):
    (pred_mu, pred_logvar), mu, logvar = model.apply(
        {"params": params},
        inputs,
        padding_mask,
        rng,
        deterministic=False,
        rngs={"dropout": rng},
    )
    pred_logvar = jnp.clip(pred_logvar, -20.0, 20.0)
    logvar = jnp.clip(logvar, -20.0, 20.0)

    mask_expanded = jnp.expand_dims(padding_mask, axis=-1)
    nll = 0.5 * (jnp.exp(-pred_logvar) * ((inputs - pred_mu) ** 2) + pred_logvar)
    sum_nll = jnp.sum(nll * mask_expanded, axis=(1, 2))
    count_valid = jnp.clip(jnp.sum(mask_expanded, axis=(1, 2)), 1e-9)
    recon_loss = jnp.mean(sum_nll / count_valid)

    kl_per_dim = -0.5 * (1 + logvar - jnp.square(mu) - jnp.exp(logvar))
    kl_loss = jnp.mean(jnp.sum(jnp.maximum(kl_per_dim, free_bits), axis=1))

    return recon_loss + beta * kl_loss, (recon_loss, kl_loss)


class FlaxSpotManager:

    def __init__(
        self,
        seq_len=4000,
        patch_size=16,
        d_model=128,
        latent_dim=64,
        memory_num_tokens=128,
        num_layers=3,
        learning_rate=1e-4,
        weight_decay=1e-4,
        max_grad_norm=1.0,
        seed=42,
    ):
        self.seq_len = seq_len
        self.latent_dim = latent_dim
        self.rng = jax.random.PRNGKey(seed)

        self.model = AstroconformerVAE(
            seq_len=seq_len,
            patch_size=patch_size,
            d_model=d_model,
            latent_dim=latent_dim,
            memory_num_tokens=memory_num_tokens,
            num_layers=num_layers,
        )

        dummy_flux = jnp.ones((1, seq_len, 1), dtype=jnp.float32)
        dummy_mask = jnp.ones((1, seq_len), dtype=jnp.float32)
        self.rng, init_rng = jax.random.split(self.rng)
        init_params = self.model.init(
            {"params": init_rng}, dummy_flux, dummy_mask, init_rng, deterministic=True
        )["params"]

        tx = optax.chain(
            optax.clip_by_global_norm(max_grad_norm),
            optax.adamw(learning_rate=learning_rate, weight_decay=weight_decay),
        )
        self.state = train_state.TrainState.create(
            apply_fn=self.model.apply, params=init_params, tx=tx
        )

        self._train_step = jax.jit(self._make_train_step())
        self._encode_step = jax.jit(
            lambda params, x, mask: self.model.apply(
                {"params": params}, x, mask, deterministic=True, method=self.model.encode
            )
        )
        self._decode_step = jax.jit(
            lambda params, z: self.model.apply(
                {"params": params}, z, deterministic=True, method=self.model.decode
            )
        )

    def _make_train_step(self):
        def train_step(state, inputs, padding_mask, rng, beta):
            rng, step_rng = jax.random.split(rng)

            def compute_loss(params):
                return sota_loss_fn(
                    params, self.model, inputs, padding_mask, step_rng, beta=beta
                )

            (loss, aux), grads = jax.value_and_grad(compute_loss, has_aux=True)(
                state.params
            )
            state = state.apply_gradients(grads=grads)
            return state, loss, aux[0], aux[1], rng

        return train_step

    def fit(self, data_generator_fn, num_epochs=10, annealing_epochs=5.0, log_interval=50):
        history = {"loss": [], "recon": [], "kl": []}

        for epoch in range(1, num_epochs + 1):
            current_beta = jnp.array(min(1.0, epoch / annealing_epochs))
            epoch_losses, epoch_recon, epoch_kl = [], [], []
            step = 0

            for batch_flux, batch_mask in data_generator_fn():
                jax_flux = jnp.array(batch_flux)
                jax_mask = jnp.array(batch_mask)

                self.state, loss, r_loss, k_loss, self.rng = self._train_step(
                    self.state, jax_flux, jax_mask, self.rng, current_beta
                )
                epoch_losses.append(loss)
                epoch_recon.append(r_loss)
                epoch_kl.append(k_loss)
                step += 1

                if log_interval and step % log_interval == 0:
                    print(
                        f"  Step {step:04d} | Loss: {loss:.4f} | Recon: {r_loss:.4f} | KL: {k_loss:.4f}"
                    )

            avg_loss = float(np.mean(epoch_losses))
            avg_recon = float(np.mean(epoch_recon))
            avg_kl = float(np.mean(epoch_kl))
            history["loss"].append(avg_loss)
            history["recon"].append(avg_recon)
            history["kl"].append(avg_kl)

            print(
                f"Epoch {epoch:02d} | Avg Loss: {avg_loss:.4f} | Avg Recon: {avg_recon:.4f} | Avg KL: {avg_kl:.4f} | Beta: {current_beta:.1f}"
            )

        return history

    def predict_segments(self, flux_segments, mask_segments=None, batch_size=8):
        num_segments = len(flux_segments)
        if mask_segments is None:
            mask_segments = np.ones((num_segments, self.seq_len), dtype=np.float32)

        latents_list = []
        recon_list = []

        for i in range(0, num_segments, batch_size):
            b_flux = jnp.array(flux_segments[i : i + batch_size])
            b_mask = jnp.array(mask_segments[i : i + batch_size])

            mu, _ = self._encode_step(self.state.params, b_flux, b_mask)
            pred_mu, _ = self._decode_step(self.state.params, mu)

            latents_list.append(np.array(mu))
            recon_list.append(np.array(pred_mu).squeeze(-1))

        return np.concatenate(latents_list, axis=0), np.concatenate(recon_list, axis=0)
