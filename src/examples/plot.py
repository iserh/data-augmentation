import vae
from utils.data import get_dataset
from utils.visualization import plot_images, plot_points
from vae import VAEForDataAugmentation, VAEConfig
from pathlib import Path


DATASET = "MNIST"
vae.models.base.model_store = f"pretrained_models/{DATASET}/all_data_single"
dataset = get_dataset(DATASET, train=False)
filepath = Path(f"img/{DATASET}/beta_test")

for e in range(5, 51, 5):
    vae_config = VAEConfig(z_dim=2, beta=0.001, attr={"mix": False, "multi_vae": False})
    model = VAEForDataAugmentation.from_pretrained(vae_config, epochs=e)

    latents, _, labels = model.encode_dataset(dataset).tensors

    (filepath / str(vae_config.beta)).mkdir(exist_ok=True, parents=True)
    plot_points(latents, labels=labels, filename=filepath / str(vae_config.beta) / f"epoch={e}.pdf", xlim=(-6, 6), ylim=(-6, 6))


# DATASET = "MNIST"
# vae.models.base.model_store = f"pretrained_models/{DATASET}/all_data_single"
# dataset = get_dataset(DATASET, train=False)
# filepath = Path(f"img/{DATASET}/beta_test")

# for beta in [1.0, 0.5, 0.1, 0.001]:
#     vae_config = VAEConfig(z_dim=2, beta=beta, attr={"mix": False, "multi_vae": False})
#     model = VAEForDataAugmentation.from_pretrained(vae_config, epochs=50)

#     latents, _, labels = model.encode_dataset(dataset).tensors

#     (filepath / str(vae_config.beta)).mkdir(exist_ok=True, parents=True)
#     plot_points(latents, labels=labels, filename=filepath / f"beta={beta}.pdf", xlim=(-6, 6), ylim=(-6, 6))
