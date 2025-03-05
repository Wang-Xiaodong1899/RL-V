import datasets as hf_datasets
import json
import openai
import pandas as pd
import time
from tqdm import tqdm

NUM_SECONDS_TO_SLEEP = 0.5

openai.api_key = 'sk-UYqwq36Z0hmfyaWJ69F675A344D645D79c9dB863Ae870eAd'
openai.base_url = "https://api.ai-gaochao.cn/v1/"

openai.default_headers = {"x-foo": "true"}


prompt = """
Give you a question and an answer.
Break the answer down into multi-step reasoning based on the question. Make sure the reasoning process and results are consistent with the information in the original answer. The number of inference steps does not exceed 4.
The format of the response is:
### Response:
Let's thick step by step.
Step 1:
Step 2:
Step 3:
...
Conclusion:

The input is:
Question: {query}

Reference Answer: {answer}

Remember: 
Do not directly answer the question in the Step 1!
the detailed conclusion must given in the last step.
Don't say any word about "reference answer", "description", "statement", "mention", "describe" in your each step.
The response is:
"""


def warp_long_sentence(query, answer):
    try:
        completion = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                "role": "user",
                "content": prompt.format(query=query, answer=answer)
                },
                ],
        )

        response = completion.choices[0].message.content
    except:
        time.sleep(NUM_SECONDS_TO_SLEEP)
        completion = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                "role": "user",
                "content": prompt.format(query=query, answer=answer)
                },
                ],
        )

        response = completion.choices[0].message.content
    
    return response

# query = "Which teddy bear is for girls?"
# answer = "The teddy bear on the left is for girls. They are often designed with feminine colors or patterns, and this one has a pink color scheme, which is typically associated with female toys."

# result = warp_long_sentence(query=query, answer=answer)
# print(result)

if __name__ == "__main__":
    data_dir = "/home/user/wangxd/RL-V/RLAIF-V-Dataset"
    data = hf_datasets.load_dataset(data_dir)['train'].cast_column("image", hf_datasets.Image(decode=False))
    
    start_row = 0
    end_row = 10000
    num_rows = end_row - start_row
    filename = f"RLAIF-V-HIERAR-{start_row}_{end_row}.jsonl"
    with open(filename, 'w', encoding='utf-8') as file:
        for idx in tqdm(range(len(data))[start_row: end_row]):
            question = data[idx]["question"]
            chosen = data[idx]["chosen"]
            rejected = data[idx]["rejected"]
            new_chosen = warp_long_sentence(query=question, answer=chosen)
            new_rejected = warp_long_sentence(query=question, answer=rejected)
            data[idx]["chosen"] = new_chosen
            data[idx]["rejected"] = new_rejected
            
            json_record = {
                    "idx": idx,
                    "question": question,
                    "chosen": chosen,
                    "rejected": rejected,
                    "new_chosen": new_chosen,
                    "new_rejected": new_rejected
                }
            file.write(json.dumps(json_record, ensure_ascii=False)+"\n")
            file.flush()