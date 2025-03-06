from muffin.data.datasets import POVIDDataset

data_dir = "/workspace/wxd/RL-V/POVID-Dataset_logps"
dataset = POVIDDataset(data_dir, None, None)
print(f"column names: {dataset.data.column_names}")