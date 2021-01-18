"""Model loading."""
import mlflow

from vae.model import VariationalAutoencoder


def get_pretrained_model(epoch: int, **hparams) -> VariationalAutoencoder:
    """Loads the pytorch VAE model and mlflow run.

    Args:
        epoch: Epoch index of checkpoint
        **hparams: VAE Hyperparameter

    Returns:
        VariationalAutoencoder: VAE
    """
    experiment_id = mlflow.get_experiment_by_name("VAE Training").experiment_id
    filter_str = (
        f"params.EPOCHS = '{hparams['EPOCHS']}' AND "
        + f"params.Z_DIM = '{hparams['Z_DIM']}' AND "
        + f"params.BETA = '{hparams['BETA']}'"
    )
    try:
        run = mlflow.search_runs(experiment_ids=[experiment_id], filter_string=filter_str).iloc[0]
    except IndexError:
        print("No run with specified criteria found!")
        exit(1)

    return mlflow.pytorch.load_model(
        run.artifact_uri + f"/model-epoch={epoch}",
    )


if __name__ == "__main__":
    model = get_pretrained_model(50, EPOCHS=250, Z_DIM=128, BETA=1.0)
    print(model)
