import json
from tqdm import tqdm

input_files = ["hieracaps_0_1000.jsonl", "hieracaps_1000_6000.jsonl", "hieracaps_6000_12818.jsonl", "hieracaps_12818_16000.jsonl"]  # Add all your input file names here
output_file = "hieracaps_ds_16000.jsonl"

with open(output_file, 'w', encoding='utf-8') as outfile:
    idx = 0
    for input_file in input_files:
        with open(input_file, 'r', encoding='utf-8') as infile:
            for line in tqdm(infile, desc=f"Processing {input_file}"):
                data = json.loads(line.strip())
                caption = data.get("caption").strip()
                chosen = data.get("chosen")
                rejected = data.get("rejected")
                ques = chosen.split("Question:")[-1].split("Response")[0].split("###")[0].strip()
                ques = f"The short image description: {caption}\n" + ques
                chos_response = chosen.split("Response:")[-1].strip()
                rejc_response = rejected.split("Response:")[-1].strip()
                new_data = {
                    "ds_name": "gpt_hierar",
                    "image": None,
                    "question": ques,
                    "chosen": chos_response,
                    "rejected": rejc_response,
                    "origin_dataset": "CC",
                    "origin_split": "train",
                    "idx": str(idx),
                    "image_path": ""
                }
                outfile.write(json.dumps(new_data)+'\n')
                idx += 1
