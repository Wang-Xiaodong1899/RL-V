import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification

import datasets as hf_datasets
import json
import os
import openai
import pandas as pd
import time
from tqdm import tqdm
import numpy as np
import fire

model_path = '/workspace/wxd/RL-V/roberta-large-wanli'
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

def process_data(mode='fine'):
    import json
    json_paths = [
        "/workspace/wxd/RL-V/CSR/LLaVA_1.5_7b_2iteration.json"
    ]
    hf_data = []
    with open(json_paths[0], 'r') as f:
        data = json.load(f)
        hf_data = hf_data + data
    filename = "/workspace/wxd/RL-V/CSR/LLaVA_1.5_7b_2iteration_Entail_Fine.jsonl"
    
    with open(filename, 'w', encoding='utf-8') as file:
        for idx in tqdm(range(len(hf_data))):
            chosen = hf_data[idx]["conversations"][1]["value"]
            rejected = hf_data[idx]["rejected_conversations"][1]["value"]
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
            hf_data[idx]["score"] = score
            file.write(json.dumps(hf_data[idx], ensure_ascii=False)+"\n")
            file.flush()

if __name__ == "__main__":
    fire.Fire(process_data)