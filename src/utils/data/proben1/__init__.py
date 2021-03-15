from .load_dataset import load_proben1

proben1_dataset_loaders = {
    "thyroid": lambda *args, **kwargs: load_proben1(dataset_name="thyroid", *args, **kwargs),
    "diabetes": lambda *args, **kwargs: load_proben1(dataset_name="diabetes", *args, **kwargs),
}
