import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


def plot_training_history(history):
    plt.figure(figsize=(10, 4))
    plt.plot(history["loss"], label="Total Loss", marker="o")
    plt.plot(history["recon"], label="Reconstruction Loss", marker="s")
    plt.title("Training History on Kepler Data")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()


def plot_latent_space(latents, labels=None, title="t-SNE Latent Space"):
    num_samples = latents.shape[0]
    if num_samples <= 5:
        print(f"サンプル数が少ない({num_samples})ため、t-SNEをスキップします。")
        return

    perplexity = min(30, num_samples - 1)
    reducer = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    latents_2d = reducer.fit_transform(latents)

    plt.figure(figsize=(8, 6))
    if labels is None:
        labels = np.arange(num_samples)
    scatter = plt.scatter(
        latents_2d[:, 0], latents_2d[:, 1], c=labels, cmap="plasma", alpha=0.8, s=60
    )
    plt.colorbar(scatter, label="Segment / Star Index")
    plt.title(title, fontsize=14)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()


def plot_reconstruction(star_data, reconstructed_segments, max_plots=3):
    num_plot = min(max_plots, star_data["num_segments"])
    if num_plot == 0:
        return

    fig, axes = plt.subplots(num_plot, 1, figsize=(12, 3 * num_plot), sharex=False)
    if num_plot == 1:
        axes = [axes]

    for i in range(num_plot):
        orig = (
            star_data["flux_segments"][i, :, 0] * star_data["flux_std"]
            + star_data["flux_mean"]
        )
        recon = (
            reconstructed_segments[i] * star_data["flux_std"] + star_data["flux_mean"]
        )
        t = star_data["time_segments"][i]

        axes[i].plot(
            t, orig, label="Original PDC-SAP Flux", color="black", lw=1.0, alpha=0.6
        )
        axes[i].plot(
            t, recon, label="Reconstructed Flux (VAE)", color="crimson", lw=2.0, alpha=0.9
        )
        axes[i].set_title(
            f"KIC {star_data['kic_id']} - Segment {i+1} / {star_data['num_segments']}",
            loc="left",
        )
        axes[i].set_ylabel("Flux")
        if i == 0:
            axes[i].legend(loc="upper right")
        axes[i].grid(True, linestyle="--", alpha=0.3)

    axes[-1].set_xlabel("Time [BKJD days]", fontsize=12)
    plt.tight_layout()
    plt.show()


def plot_latent_traversal(
    manager, base_latent, target_dim=0, traversal_range=(-3.0, 3.0), num_steps=5
):
    z_values = np.linspace(traversal_range[0], traversal_range[1], num_steps)
    traversal_z = np.tile(base_latent, (num_steps, 1))
    traversal_z[:, target_dim] = z_values

    pred_mu, _ = manager._decode_step(manager.state.params, jnp.array(traversal_z))
    reconstructed_flux = np.array(pred_mu).squeeze(-1)

    fig, axes = plt.subplots(
        num_steps, 1, figsize=(10, 1.2 * num_steps), sharex=True, sharey=True
    )
    for i in range(num_steps):
        axes[i].plot(reconstructed_flux[i], color="crimson", lw=1.5)
        axes[i].set_title(
            f"z_{target_dim} = {z_values[i]:.2f}", fontsize=10, loc="right"
        )
        axes[i].grid(True, linestyle="--", alpha=0.3)
        axes[i].set_ylabel("Flux")
    axes[-1].set_xlabel("Time Steps", fontsize=12)
    plt.suptitle(f"Latent Traversal of Dimension {target_dim}", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.show()
