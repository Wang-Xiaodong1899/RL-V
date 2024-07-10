import json
from tqdm import tqdm
import os

all_data = []
all_ids = []
path = "/mnt/storage/user/wangxiaodong/data/Hound-DPO/video_240k_caption_15k.jsonl"
# path = '/mnt/storage/user/wangxiaodong/data/Hound-DPO/sft_dpo_17k.jsonl'
with open(path, 'r') as f:
    for line in f:
        item = json.loads(line)
        all_data.append(item)
        all_ids.append(item['id'])
print(len(all_ids))

root_dir = "/mnt/storage/user/wangxiaodong/data/Hound-DPO/train_300k"
exist_video_names = os.listdir(root_dir)

print(f'exist video length: {len(exist_video_names)}')


filter_data = []

for ex_name in tqdm(exist_video_names):
    if ex_name in all_ids:
        filter_data.append(ex_name)

print(f'filter data length: {len(filter_data)}')

if len(filter_data) > 0:
    pass
    # mini_path = "./mini_hound_15_tar_gz.json"
    # with open(mini_path, 'w') as f:
    #     json.dump(filter_data, f)
else:
    print("no data")
