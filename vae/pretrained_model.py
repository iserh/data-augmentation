"""Model loading."""
import mlflow

from vae.model import VariationalAutoencoder


def get_pretrained_model(epoch_chkpt: int, **hparams) -> VariationalAutoencoder:
    """Loads the pytorch VAE model and mlflow run.

    Args:
        epoch_chkpt: Epoch index of checkpoint
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
        run = mlflow.search_runs(experiment_ids=[experiment_id], filter_string=filter_str)
        if 'tags.SELECT' in run:
            run = run.loc[run['tags.SELECT'] == "True"].iloc[0]
        else:
            run = run.iloc[0]
    except IndexError:
        print("No run with specified criteria found!")
        exit(1)

    return mlflow.pytorch.load_model(
        run.artifact_uri + f"/model-epoch={epoch_chkpt}",
    )


if __name__ == "__main__":
    from utils.config import mlflow_roots
    mlflow.set_tracking_uri(mlflow_roots["MNIST"])
    model = get_pretrained_model(0, EPOCHS=100, Z_DIM=2, BETA=1.0)
    print(model)
