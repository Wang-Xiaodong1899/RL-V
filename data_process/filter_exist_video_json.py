import os
import json

def load_jsonl(save_path):
    with open(save_path, "r") as f:
        data = [json.loads(line) for line in f.readlines()]
    return data

def save_jsonl(save_path, data, append=False):
    if append:
        mode = "a"
    else:
        mode = "w"
    if type(data) == list:
        with open(save_path, mode) as f:
            for line in data:
                json.dump(line, f)
                f.write("\n")
    else:
        with open(save_path, mode) as f:
            json.dump(data, f)
            f.write("\n")

videonames = os.listdir("/mnt/storage/user/wangxiaodong/RLAIF-V/data_process/dataset/train")
exsited_videos = []
for videoname in videonames:
    if videoname.endswith(".mp4"):
        exsited_videos.append(videoname.split(".mp4")[0])

print(f"valid num {len(exsited_videos)}")

dpo_data = load_jsonl("/mnt/storage/user/wangxiaodong/data/Hound-DPO/sft_dpo_17k.jsonl")
save_data = []
for item in dpo_data:
    if item["id"] in exsited_videos:
        save_data.append(item)
valid_num = len(save_data)
print(f"valid num {valid_num}")
save_jsonl(f"/mnt/storage/user/wangxiaodong/data/Hound-DPO/sft_dpo_{valid_num}.jsonl", save_data)