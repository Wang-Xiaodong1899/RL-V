import datasets as hf_datasets
import json
from openai import OpenAI
import os
import pandas as pd
from tqdm import tqdm

from functools import partial


def modify_data(item, idx, anno):
    item["chosen"] = anno[idx]["new_chosen"].split("Response:")[-1].strip()
    item["rejected"] = anno[idx]["new_rejected"].split("Response:")[-1].strip()
    return item

if __name__ == "__main__":
    data_dir = "/root/autodl-tmp/RL-V/RLAIF-V-Dataset"
    data = hf_datasets.load_dataset(data_dir)['train'].cast_column("image", hf_datasets.Image(decode=False))
    cache_file = "/root/autodl-tmp/RL-V/RLAIF-V-HIERAR-Dataset-6k"
    os.makedirs(cache_file, exist_ok=True)
    
    data = data.select(range(6000))
    input_file = "/root/autodl-tmp/RL-V/RLAIF-V-HIERAR-0_10000.jsonl"
    annot_data = []
    print(f"read jsonl from {input_file}")
    with open(input_file, 'r', encoding='utf-8') as infile:
        for line in tqdm(infile, desc=f"Processing {input_file}"):
            annot_data.append(json.loads(line.strip()))
    
    partial_modify_data = partial(modify_data, anno=annot_data)
    
    new_data = data.map(partial_modify_data, with_indices=True)
    
    print(new_data[0]["chosen"])

    df = pd.DataFrame(new_data)
    start = 0
    df.to_parquet(os.path.join(cache_file, f'RLAIF-V-HIERAR_{start:03}-{len(data)}.parquet'))
