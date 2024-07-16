import openai

openai.api_key = 'sk-UYqwq36Z0hmfyaWJ69F675A344D645D79c9dB863Ae870eAd'
openai.base_url = "https://api.ai-gaochao.cn/v1/"

openai.default_headers = {"x-foo": "true"}

prompt = """
Give you a long sentence composed of 4 sentences with hierarchical relationship, each sentence is separated by =>, each sentence is a model description of the image concept, and 4 sentences show the reasoning process. Please design a problem, which basically means that the model can deduce the result step by step from the picture, and rewrite 4 sentences into 4 steps, such as:
Q:
A:
Let's thick step by step.
Step 1:
Step 2:
Step 3:
Step 4:
The input long sentence is:
{query}
The response is:
"""

# input = "cuteness => baby => cute baby sitting in a high chair => cute baby sitting in a high chair waiting for dinner"
query = "wood => firewood => firewood on the ground => firewood from the sawed pine trees lie on the ground"

completion = openai.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {
        "role": "user",
        "content": prompt.format(query=query)
        },
        ],
)

print(completion.choices[0].message.content)
# print(completion.model)