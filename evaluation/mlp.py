from typing import Optional
from torch.utils.data import ConcatDataset, DataLoader
from torch.utils.data import TensorDataset
from utils.integrations import BackendStore, ExperimentName
from vae import VAEBaseModel, VAEConfig, VAEForDataAugmentation, Interpolation, Noise
from sklearn.model_selection import train_test_split
from utils.data import Datasets
import mlflow
from evaluation.models import MLP
from evaluation.trainer import Trainer, TrainingArguments
from enum import Enum
from dataclasses import fields


class AugmentationMethods(Enum):
    INTERPOLATION: str = "interpolation"
    EXTRAPOLATION: str = "extrapolation"
    NOISE: str = "noise"
    NOISE_EXTRA: str = "noise_extra"
    NOISE_INTER: str = "noise_inter"
    BASELINE: str = "baseline"


def train_mlp(
    dataset_name: str,
    dataset_size: int,
    vae_config: Optional[VAEConfig] = None,
    augmentation_method: AugmentationMethods = AugmentationMethods.BASELINE,
    alpha: Optional[float] = None,
    alpha2: Optional[float] = None,
    k: Optional[int] = None,
    k2: Optional[int] = None,
) -> None:
    # initialize mlflow experiment
    mlflow.set_tracking_uri(BackendStore[dataset_name].value)
    mlflow.set_experiment(ExperimentName.MLP.value)

    # *** data preparation ***

    # evaluation dataset
    eval_dataset = Datasets(dataset_name, train=False)
    # the full dataset partition train
    train_dataset_full = Datasets(dataset_name, train=True)
    # extract the image and target tensors
    images, targets = next(iter(DataLoader(train_dataset_full, batch_size=len(train_dataset_full))))

    # split dataset into train and validation parts
    x_train, x_eval, y_train, y_eval = train_test_split(
        images,
        targets,
        stratify=targets,
        shuffle=True,
        train_size=dataset_size / len(train_dataset_full),
        test_size=5000 / len(train_dataset_full),
    )

    # recreate torch datasets
    train_dataset = TensorDataset(x_train, y_train)
    test_dataset = TensorDataset(x_eval, y_eval)

    print("Test:", len(test_dataset))
    print("Eval:", len(eval_dataset))
    print("Train:", len(train_dataset))

    if augmentation_method == AugmentationMethods.BASELINE:
        train_dataset_final = train_dataset
    else:
        # load vae model
        vae_model = VAEBaseModel.from_pretrained(vae_config)
        # create vae for data augmentation
        vae = VAEForDataAugmentation(vae_model)

        # *** data generation ***

        # encode the train dataset
        latents, targets = vae.encode_dataset(train_dataset).tensors

        if augmentation_method == AugmentationMethods.INTERPOLATION:
            # augment using interpolation
            interpolation = Interpolation(alpha=alpha, k=k)
            latents, targets = interpolation(latents, targets)
        elif augmentation_method == AugmentationMethods.EXTRAPOLATION:
            # augment using extrapolation
            interpolation = Interpolation(alpha=-alpha, k=k)
            latents, targets = interpolation(latents, targets)
        elif augmentation_method == AugmentationMethods.NOISE:
            # augment using noise
            noise = Noise(alpha=alpha, k=k, std=latents.std())
            latents, targets = noise(latents, targets)
        elif augmentation_method == AugmentationMethods.NOISE_INTER:
            # augment using interpolation and noise
            interpolation = Interpolation(alpha=alpha, k=k)
            latents, targets = interpolation(latents, targets)
            noise = Noise(alpha=alpha2, k=k2, std=latents.std())
            latents, targets = noise(latents, targets)
        elif augmentation_method == AugmentationMethods.NOISE_EXTRA:
            # augment using interpolation and noise
            interpolation = Interpolation(alpha=-alpha, k=k)
            latents, targets = interpolation(latents, targets)
            noise = Noise(alpha=alpha2, k=k2, std=latents.std())
            latents, targets = noise(latents, targets)

        # decode augmented latents
        decoded = vae.decode_dataset(TensorDataset(latents, targets))
        # concat datasets
        train_dataset_final = ConcatDataset([train_dataset, decoded])
        print("Train (augmented):", len(train_dataset_final))

    # create mlp model
    mlp = MLP()
    # training arguments for trainer
    training_args = TrainingArguments(
        epochs=20,
        save_intervall=None,
        save_model=False,
    )
    # trainer
    trainer = Trainer(
        args=training_args,
        model=mlp,
        train_dataset=train_dataset_final,
        test_dataset=test_dataset,
        eval_dataset=eval_dataset,
    )

    # training
    with mlflow.start_run(run_name=augmentation_method.value):
        # log params
        if augmentation_method != AugmentationMethods.BASELINE:
            mlflow.log_params({"vae_" + f.name: getattr(vae_config, f.name) for f in fields(vae_config)})
            mlflow.log_params({
                "alpha_interpolation": alpha,
                "alpha_noise": alpha2,
                "k_interpolation": k,
                "k_noise": k2,
                "dataset_size": dataset_size
            })

        # train model
        trainer.train()
        # evaluate model
        trainer.evaluate()


augmentations = [
    {
        "augmentation_method": AugmentationMethods.BASELINE,
        "alpha": None,
        "alpha2": None,
        "k": None,
        "k2": None,
    },
    {
        "augmentation_method": AugmentationMethods.INTERPOLATION,
        "alpha": 0.5,
        "alpha2": None,
        "k": 3,
        "k2": None,
    },
    {
        "augmentation_method": AugmentationMethods.EXTRAPOLATION,
        "alpha": 0.5,
        "alpha2": None,
        "k": 3,
        "k2": None,
    },
    {
        "augmentation_method": AugmentationMethods.NOISE,
        "alpha": 0.5,
        "alpha2": None,
        "k": 1,
        "k2": None,
    },
    {
        "augmentation_method": AugmentationMethods.NOISE_INTER,
        "alpha": 0.5,
        "alpha2": 0.3,
        "k": 3,
        "k2": 1,
    },
    {
        "augmentation_method": AugmentationMethods.NOISE_EXTRA,
        "alpha": 0.5,
        "alpha2": 0.3,
        "k": 3,
        "k2": 1,
    },
]

vae_configurations = [
    VAEConfig(epochs=100, checkpoint=100, z_dim=10, beta=1.0),
    VAEConfig(epochs=100, checkpoint=100, z_dim=20, beta=1.0),
    VAEConfig(epochs=100, checkpoint=100, z_dim=50, beta=1.0),
    VAEConfig(epochs=100, checkpoint=100, z_dim=100, beta=1.0),
]


if __name__ == "__main__":
    n_trials = 3

    for vae_config in vae_configurations:
        for aug in augmentations:
            for _ in range(n_trials):
                train_mlp(
                    dataset_name="MNIST",
                    dataset_size=1000,
                    vae_config=vae_config,
                    **aug,
                )
