import datasets as hf_datasets

# hf_data = hf_datasets.load_dataset("parquet", data_files='/workspace/wxd/RL-V/RLHF-V-Dataset/RLHF-V-Dataset.parquet')['train'].cast_column("image", hf_datasets.Image(decode=False))
# hf_data = hf_datasets.load_dataset("/workspace/wxd/RL-V/RLHF-V-Dataset")['train'].cast_column("image", hf_datasets.Image(decode=False))

# print(type(hf_data))

# data_dir = "/workspace/wxd/RL-V/RLHF-V-Dataset_logps"

# data_dir = "/workspace/wxd/RL-V/RLAIF-V-Dataset"
# data = hf_datasets.load_dataset(data_dir)['train'].cast_column("image", hf_datasets.Image(decode=False))


data = hf_datasets.load_dataset("/workspace/wxd/RL-V/RLAIF-V-HIERAR-Dataset-6k")['train'].cast_column("image", hf_datasets.Image(decode=False))


print(len(data))
sample = data[0]
print(sample['image']["bytes"])
print(sample['ds_name'])
print(sample['image']["path"])
print(sample['image_path'])
print(sample.keys())

