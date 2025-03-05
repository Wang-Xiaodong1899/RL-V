import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification

import datasets as hf_datasets
import json
import openai
import pandas as pd
import time
from tqdm import tqdm
import numpy as np
import fire

model_path = '/home/user/wangxd/RL-V/roberta-large-wanli'
model = RobertaForSequenceClassification.from_pretrained(model_path)
tokenizer = RobertaTokenizer.from_pretrained(model_path)

model.to("cuda:0")


print(f"loaded models")

def predict(sentence1, sentence2):
    inputs = tokenizer(sentence1, sentence2, return_tensors='pt', max_length=128, truncation=True)
    inputs = inputs.to("cuda:0")
    logits = model(**inputs).logits
    probs = logits.softmax(dim=1).squeeze(0)
    # print(probs) # [contradiction, entailment, neutral]
    # label_id = torch.argmax(probs).item()
    # prediction = model.config.id2label[label_id]
    return list(probs.detach().cpu().numpy().astype(np.float64))

# chosen = "The image depicts a train traveling on a track through a countryside setting with tall grass, trees, and power lines in the background."
# rejected = "The image is set in an open area with train tracks, grassy fields, and trees in the background."

# predict(chosen, rejected)

def process_data(mode='fine', num=10000, data_dir="/home/user/wangxd/RL-V/RLAIF-V-Dataset"):
    data = hf_datasets.load_dataset(data_dir)['train'].cast_column("image", hf_datasets.Image(decode=False))
    
    start_row = 0
    end_row = len(data)
    num_rows = end_row - start_row
    filename = f"RLAIF-V-Entail-{mode}-{start_row}_{end_row}.jsonl"
    with open(filename, 'w', encoding='utf-8') as file:
        for idx in tqdm(range(len(data))[start_row: end_row]):
            question = data[idx]["question"]
            chosen = data[idx]["chosen"]
            rejected = data[idx]["rejected"]
            if mode == 'fine':
                scores = []
                rejected_segs = rejected.split('. ')
                for seg in rejected_segs:
                    score = predict(chosen, seg)
                    scores.append(score)
                scores = np.array(scores)
                score = list(scores.mean(0))
            else:
                score = predict(chosen, rejected)
            # print(score, type(score))
            json_record = {
                    "idx": idx,
                    "question": question,
                    "chosen": chosen,
                    "rejected": rejected,
                    "score": score
                }
            file.write(json.dumps(json_record, ensure_ascii=False)+"\n")
            file.flush()

if __name__ == "__main__":
    fire.Fire(process_data)