import openai

openai.api_key = 'sk-UYqwq36Z0hmfyaWJ69F675A344D645D79c9dB863Ae870eAd'
openai.base_url = "https://api.ai-gaochao.cn/v1/"

openai.default_headers = {"x-foo": "true"}

prompt = """
You are given a question and a long answer, rearrange the long answer according to the question in a strict way of reasoning step by step and finally reaching a conclusion.
Each step needs to have enough information, please minimize the number of reasoning steps, no more than 4 steps.
Finally, answer the question briefly.
Follow the format:
"
step 1:
...
step n:
The brief answer is:
"

Remember, don't mention in your response that I gave you the answer, assuming it's image information that you know.

Question: "{ques}"
Answer: "{ans}"

Your response is:
"""


ques = "Where are the man and his dog located in this image?"
ans = "The man and his dog are located on the beach, near the ocean. They are either jogging or walking along the shoreline, where the land meets the water."

completion = openai.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {
        "role": "user",
        "content": prompt.format(ques=ques, ans=ans)
        },
        ],
)

print(completion.choices[0].message.content)
print(completion.model)