from muffin.data.datasets import SEVADataset

data_dir = "/root/autodl-tmp/RL-V/SEVA-Dataset_logps"

list_data_dict = SEVADataset(data_dir, None, None, None,None, None, is_llava15=True)

item = list_data_dict[0]
