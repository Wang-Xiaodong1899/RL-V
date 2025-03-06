import json
import os
import fire

def main(dirs=""):
    # dirs = "/root/autodl-tmp/RL-V/SeVa/seva/pope_result/root/autodl-tmp/RL-V/seva-7b-diffu500/"
    files = ['adv', 'random', 'popular']
    all_data = []
    for file in files:
        with open(os.path.join(dirs, f"pope_{file}.jsonl"), 'r', encoding='utf-8') as f:
            for line in f:
                json_obj = json.loads(line)
                
                json_obj["text"] = json_obj.pop("answer")
                if file == 'random':
                    json_obj["question_id"] = json_obj["question_id"] + 10000000
                    json_obj["category"] = "random"
                elif file == 'popular':
                    json_obj["question_id"] = json_obj["question_id"] + 20000000
                    json_obj["category"] = "popular"
                else:
                    pass
                    json_obj["category"] = "adversarial"
                all_data.append(json_obj)

    with open(os.path.join(dirs, f"pope_all.jsonl"), 'w', encoding='utf-8') as f:
        for item in all_data:
            f.write(json.dumps(item) + '\n')

if __name__ == "__main__":
    fire.Fire(main)