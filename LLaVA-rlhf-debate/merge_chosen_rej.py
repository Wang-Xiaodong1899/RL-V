import json

# 读取jsonl文件
def read_jsonl(file_path):
    with open(file_path, 'r') as f:
        return [json.loads(line) for line in f]

# 按照id合并两个文件，仅保留rejected_file.jsonl中存在的id
def merge_jsonl(file1_path, file2_path, output_path):
    data1 = read_jsonl(file1_path)  # 读取chosen文件
    data2 = read_jsonl(file2_path)  # 读取rejected文件

    # 使用字典按id保存chosen文件的数据
    chosen_data = {item['id']: item for item in data1}

    # 合并数据，只保留在rejected文件中有的id
    merged_data = []
    for item in data2:
        rejected_id = item['id']
        if rejected_id in chosen_data:
            merged_item = {**chosen_data[rejected_id], 'rejected': item['rejected']}
            merged_data.append(merged_item)

    # 将合并后的数据写入新的jsonl文件
    with open(output_path, 'w') as f:
        for line in merged_data:
            f.write(json.dumps(line) + '\n')

# 示例使用
merge_jsonl('/workspace/wxd/RL-V/LLaVA-rlhf-debate/llava_rlhf_for_leanpo-1w.jsonl', '/workspace/wxd/RL-V/LLaVA-rlhf-aug/llava_rlhf_aug-1w.jsonl', './llava_rlhf_merge-for_leanpo-1w.jsonl')
