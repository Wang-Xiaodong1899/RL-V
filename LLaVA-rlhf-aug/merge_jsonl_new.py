import json
import os

def merge_jsonl_files(file_list, output_file):
    """
    按顺序合并多个 jsonl 文件到一个新的 jsonl 文件

    :param file_list: 文件名列表，按顺序合并
    :param output_file: 输出的合并后的文件名
    """
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for file_name in file_list:
            with open(file_name, 'r', encoding='utf-8') as infile:
                for line in infile:
                    
                    outfile.write(line)

root = "/home/user/wangxd/RL-V/LLaVA-rlhf-aug"

file_list = ['llava_rlhf_for_leanpo_0_2500.jsonl', 'llava_rlhf_for_leanpo_2500_5000.jsonl', 'llava_rlhf_for_leanpo_5000_7500.jsonl', 'llava_rlhf_for_leanpo_7500_10000.jsonl']

file_list = [os.path.join(root, f) for f in file_list]

output_file = 'llava_rlhf_aug-1w.jsonl'

output_file = os.path.join(root, output_file)

merge_jsonl_files(file_list, output_file)

print(f"文件已合并到 {output_file}")