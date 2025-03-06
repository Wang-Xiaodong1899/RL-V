import json

import torch
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from PIL import Image
import base64
import io
import os
from omnilmm.model.omnilmm import OmniLMMForCausalLM
from omnilmm.model.utils import build_transform
from omnilmm.train.train_utils import omni_preprocess
from transformers import AutoTokenizer, AutoModel

import fire
from tqdm import tqdm

DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"


def expand_question_into_multimodal(question_text, image_token_len, im_st_token, im_ed_token, im_patch_token):
    if '<image>' in question_text[0]['content']:
        question_text[0]['content'] = question_text[0]['content'].replace(
            '<image>', im_st_token + im_patch_token * image_token_len + im_ed_token)
    else:
        question_text[0]['content'] = im_st_token + im_patch_token * \
            image_token_len + im_ed_token + '\n' + question_text[0]['content']
    return question_text


def img2base64(file_name):
    with open(file_name, 'rb') as f:
        encoded_string = base64.b64encode(f.read())
        return encoded_string

class RLAIFV7B:
    def __init__(self, model_path) -> None:
        disable_torch_init()
        model_name='llava-v1.5-7b'
        tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, model_base=None,model_name=model_name, device_map={"": 'cuda'})
        print(f'loaded pretrained model from {model_path}')
        self.tokenizer=tokenizer
        self.model=model
        self.image_processor=image_processor
        self.context_len=context_len

    def chat(self, input):
        msgs = input['question']
        if self.model.config.mm_use_im_start_end:
            msgs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + msgs
        else:
            msgs = DEFAULT_IMAGE_TOKEN + '\n' + msgs
        
        msgs = input.get("prefix", "") + msgs
        # print(msgs)

        image = Image.open(input['image']).convert('RGB')
        conv = conv_templates["llava_v1"].copy()
        conv.append_message(conv.roles[0], msgs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
        image_tensor = process_images([image], self.image_processor, self.model.config)[0]
        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=image_tensor.unsqueeze(0).half().cuda(),
                image_sizes=[image.size],
                do_sample=True,
                temperature=0.2,
                max_new_tokens=512,
                use_cache=True)
        outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        return outputs


class RLAIFVChat:
    def __init__(self, model_path) -> None:
        if '12B' in model_path:
            pass
        else:
            self.model = RLAIFV7B(model_path)

    def chat(self, input):
        return self.model.chat(input)

# initial chat_model
chat_model = RLAIFVChat(model_path="/root/autodl-fs/llava-v1.5-7b")

def inference_pipeline(start=0, end=10000):
    # ************* read dataset *************

    with open('/data/wangxd/llava-critic-113k/LLaVA-Human-Preference-10K/llava_7b_v1_preference.json', 'r') as f:
        llava_rlhf_data = json.load(f)

    COCO_ROOT = "/data/wangxd/mscoco/train2014"
    
    new_samples = []
    
    with open(f'/root/autodl-tmp/RL-V/LLaVA-rlhf-debate/llava_rlhf_for_leanpo_{start}_{end}.jsonl', 'w', encoding='utf-8') as f:
        for idx, sample in tqdm(enumerate(llava_rlhf_data[start: end])):
            model_return = None
            for turn in range(1):
                image_path = sample["image"] # 000000XXX.jpg
                image_id = sample["id"]
                questions = [convo['value'] for convo in sample['conversations'] if convo['from'] == 'human']
                answers = [convo['value'] for convo in sample['conversations'] if convo['from'] == 'gpt']

                # conv1
                question = questions[0]
                answer = answers[0]
                
                prompt = question
                gt_answer = answer
                
                question = question.replace('<image>', '').replace('\n', '')
                
                prompt = prompt.replace('<image>', '').replace('\n', '')
                qs = prompt
                
                if turn == 0:
                    prefix = "Here are some hints: " + answer + "\n\n" + "Please respond based on the given hints and image content." + "\n" + "Please answer in your own way and enrich your answer." + "\n\n"
                else:
                    prefix = f"""
Your previous reply to me was:
{model_return}. This response can continue to be improved.

Now, please align your response with the information below:
{answer}

You need to reflect the given information as best you can, optimize your response, and enrich your answer. I'll ask you the question again:

"""
                image_path = os.path.join(COCO_ROOT, "COCO_train2014_"+ image_path)
                
                # msgs = prefix + "\n\n" + qs
                
                inputs = {"image": image_path, "prefix": prefix, "question": qs}
                
                previous_return = model_return if model_return else ""
                
                model_return = chat_model.chat(inputs)
                
                # print(model_return)
    
            json_line = {
                'id': sample['id'],
                'image': sample['image'],
                'prompt': prompt,
                'answer': gt_answer,
                # 'first': previous_return,
                'chosen': model_return,
                # "rejected": sample["rejected"]
            }
            f.write(json.dumps(json_line, ensure_ascii=False) + '\n')
            f.flush()
    
    print(f'inference {start} to {end} done!')
    
if __name__ == '__main__':
    fire.Fire(inference_pipeline)
