import openai
import time

NUM_SECONDS_TO_SLEEP = 0.5

class Chat:
    def __init__(self, model="", timeout_sec=20, openai_apikey=''):
        self.model = "gpt-3.5-turbo"
        self.timeout = timeout_sec
        # openai.api_key = openai_apikey
        openai.api_key = 'sk-UYqwq36Z0hmfyaWJ69F675A344D645D79c9dB863Ae870eAd'
        openai.base_url = "https://api.ai-gaochao.cn/v1/"
        # openai.default_headers = {"x-foo": "true"}

    def chat_completion(self, messages, temperature=0.2, top_p=1, max_tokens=512,
                        presence_penalty=0, frequency_penalty=0):

        response = openai.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty
        )

        return response


# 'gpt-4-0314'
def get_eval(model, content: str,
             chat_gpt_system='You are a helpful and precise assistant for checking the quality of the answer.',
             max_tokens: int=256,
             fail_limit=100,
             openai_apikey="sk-LBkpGaBchYZDL2bLE5A9Af69E93c4365A7De76F4F5F0EaD4"):

    openai.api_key = openai_apikey
    # openai.api_key = 'sk-LBkpGaBchYZDL2bLE5A9Af69E93c4365A7De76F4F5F0EaD4'
    openai.base_url = "http://ideal-lattice.com:3000/v1/"
    openai.default_headers = {"x-foo": "true"}

    while True:
        try:
            response = openai.chat.completions.create(
                model=model,
                messages=[{
                    'role': 'system',
                    'content': chat_gpt_system
                }, {
                    'role': 'user',
                    'content': content,
                }],
                temperature=0.2,
                max_tokens=max_tokens,
            )

            if response.model != model:
                real_model = response.model
                print(f'Except f{model}, but got message from f{real_model}', flush=True)
                continue

            print(response.model)

            break
        except openai.error.RateLimitError:
            pass
        except Exception as e:
            print(e)
        time.sleep(NUM_SECONDS_TO_SLEEP)

    return response.choices[0].message.content