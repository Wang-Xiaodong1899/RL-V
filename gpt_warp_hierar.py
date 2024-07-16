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
Give you a long sentence composed of 4 sentences with hierarchical relationship, each sentence is separated by =>, each sentence is a model description of the image concept, and 4 sentences show the reasoning process. Please design a problem, which basically means that the model can deduce the result step by step from the picture, and rewrite 4 sentences into 4 steps, such as:
### Question:

### Response:
Let's thick step by step.
Step 1:
Step 2:
Step 3:
Step 4:

The input is:
{query}

The question and response are:

"""


def warp_long_sentence(query):
    # input = "cuteness => baby => cute baby sitting in a high chair => cute baby sitting in a high chair waiting for dinner"
    # query = "wood => firewood => firewood on the ground => firewood from the sawed pine trees lie on the ground"
    try:
        completion = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                "role": "user",
                "content": prompt.format(query=query)
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
                "content": prompt.format(query=query)
                },
                ],
        )

        response = completion.choices[0].message.content
    
    return response

if __name__ == "__main__":
    
    # read csv
    start_row = 6000
    end_row = 16000
    num_rows = end_row - start_row
    filename = f"hieracaps_{start_row}_{end_row}.jsonl"
    df = pd.read_csv("/mnt/storage/user/wangxiaodong/RLAIF-V/hierarcaps/hierarcaps_train.csv", skiprows=range(1, start_row+1), nrows=num_rows)
    with open(filename, 'w', encoding='utf-8') as file:
        for idx, row in tqdm(df.iterrows()):
            pos_sentence = row["positive"]
            neg_sentence = row["negative"]
            caption = pos_sentence.split("=>")[3]
            chosen = warp_long_sentence(pos_sentence)
            rejected = warp_long_sentence(neg_sentence)
            json_record = {
                "caption": caption,
                "chosen": chosen,
                "rejected": rejected
            }
            file.write(json.dumps(json_record, ensure_ascii=False)+"\n")
            file.flush()
    
    
    