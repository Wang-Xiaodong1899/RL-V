import openai

openai.api_key = 'sk-LBkpGaBchYZDL2bLE5A9Af69E93c4365A7De76F4F5F0EaD4'
openai.base_url = "http://ideal-lattice.com:3000/v1/"

openai.default_headers = {"x-foo": "true"}

completion = openai.chat.completions.create(
    model="gpt-4-turbo",
    messages=[
        {
        "role": "user",
        "content": "hi",
        },
        ],
)

print(completion.choices[0].message.content)
print(completion.model)