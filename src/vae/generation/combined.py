import mlflow
from core.data import MNIST_Dataset
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader, TensorDataset

from utils.data import DataFetcher
from utils.integrations import BackendStore, ExperimentName
from vae.applied_vae import VAEForDataAugmentation
from vae.generation.interpolation import Interpolation
from vae.generation.noise import Noise
from vae.models import MNISTVAE, VAEConfig
from vae.visualization import visualize_images, visualize_latents, visualize_real_fake_images

# *** HYPERPARAMETERS ***

DATASET = "MNIST"
vae_config = VAEConfig(total_epochs=100, epochs=100, z_dim=10, beta=1.0)
N_SAMPLES = 1024
K_INTERPOLATION = 3
K_NOISE = 1
ALPHA_INTERPOLATION = 0.5
ALPHA_NOISE = 0.2

# initialize mlflow experiment
mlflow.set_tracking_uri(BackendStore[DATASET].value)
mlflow.set_experiment(ExperimentName.VAEGeneration.value)

# the dataset
dataset = MNIST_Dataset()
# load pretrained vae
model = MNISTVAE.from_pretrained(vae_config)
# applied vae framework
vae = VAEForDataAugmentation(model)

# load N_SAMPLES in batched mode
datafetcher = DataFetcher(dataset, N_SAMPLES)
loaded_dataset = datafetcher()

# create encode pipeline
encoded_fetch = vae.encode_dataset(loaded_dataset)
reals = loaded_dataset.tensors[0]
latents, targets = next(iter(DataLoader(encoded_fetch, batch_size=N_SAMPLES)))

with mlflow.start_run(run_name="combined"):
    # log params to mlflow
    mlflow.log_params(vae_config.__dict__)
    mlflow.log_params(
        {
            "ALPHA_INTERPOLATION": ALPHA_INTERPOLATION,
            "ALPHA_NOISE": ALPHA_NOISE,
            "K_INTERPOLATION": K_INTERPOLATION,
            "K_NOISE": K_NOISE,
            "N_SAMPLES": N_SAMPLES,
        }
    )

    # visualize latents
    print("Visualizing latents...")
    pca = PCA(2).fit(latents) if latents.size(1) > 2 else None
    visualize_latents(latents, pca, targets, color_by_target=True, img_name="encoded")

    # interpolate
    print("Interpolating...")
    interpolation = Interpolation(alpha=ALPHA_INTERPOLATION, k=K_INTERPOLATION, return_indices=True)
    mod_latents, mod_targets, indices = interpolation(latents, targets)
    visualize_latents(mod_latents, pca, mod_targets, color_by_target=True, img_name="interpolated")

    # add noise
    print("Adding noise...")
    noise = Noise(alpha=ALPHA_NOISE, k=K_NOISE, std=mod_latents.std())
    mod_latents, mod_targets = noise(latents, targets)
    visualize_latents(mod_latents, pca, mod_targets, color_by_target=True, img_name="with_noise")

    # decode modified latents
    fakes_set = vae.decode_dataset(TensorDataset(mod_latents, mod_targets))
    fakes = next(iter(DataLoader(fakes_set, batch_size=20 * K_INTERPOLATION * K_NOISE)))[0]

    # show real images and fake images
    print("Show Real - Fake")
    visualize_real_fake_images(reals, fakes, n=20, k=K_INTERPOLATION * K_NOISE, indices=indices)
