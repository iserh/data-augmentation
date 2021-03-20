from utils.data import load_splitted_datasets, split_datasets

DATASET = "MNIST"
split_datasets(DATASET, reduce=500, others=False, seed=42)
datasets, _ = load_splitted_datasets(DATASET, others=False)
print(", ".join([str(len(ds)) for ds in datasets]))
