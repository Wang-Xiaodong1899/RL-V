from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from transformers import TextStreamer
from io import BytesIO
from PIL import Image
import requests
import datetime
import argparse
import torch
import tqdm
import os
import json

import sys
sys.path.append("/root/autodl-tmp/RL-V/")


def main(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    # model_name = get_model_name_from_path(model_path)
    if args.model_base is None:
        model_name = "llava-v1.5-7b"
        tokenizer, model, image_processor, context_len = load_pretrained_model(
            model_path, model_base=args.model_base, model_name=model_name, device_map={"": 'cuda'})
    else:
        model_name = "llava_lora_model"
        tokenizer, model, image_processor, context_len = load_pretrained_model(
            model_path, model_base=args.model_base, model_name=model_name, device_map={"": 'cuda'})

    print(f'loaded pretrained model from {model_path}')

    image = Image.open(args.image_path)
    image_tensor = process_images([image], image_processor, model.config)[0]
    images = image_tensor.unsqueeze(0).half().cuda()
    image_sizes = [image.size]
    qs = args.qs
    cur_prompt = qs
    if getattr(model.config, 'mm_use_im_start_end', False):
        qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + \
            DEFAULT_IM_END_TOKEN + '\n' + qs
    else:
        qs = DEFAULT_IMAGE_TOKEN + '\n' + qs
    cur_prompt = '<image>' + '\n' + cur_prompt

    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    input_ids = tokenizer_image_token(
        prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=images,
            image_sizes=image_sizes,
            do_sample=True if args.temperature > 0 else False,
            temperature=args.temperature,
            max_new_tokens=1024,
            use_cache=True,
        )

    outputs = tokenizer.batch_decode(
        output_ids, skip_special_tokens=True)[0].strip()

    print(outputs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str,
                        default="/root/autodl-tmp/RL-V/seva-7b-diffu500")
    parser.add_argument("--model-base", type=str,
                        default="/root/autodl-fs/llava-v1.5-7b")
    parser.add_argument("--image-path", type=str, default="/root/autodl-tmp/RL-V/examples/test.jpeg")
    parser.add_argument(
        "--qs", type=str, default="Describe this image briefly.")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--answer-prompter", action="store_true")
    parser.add_argument("--single-pred-prompt", action="store_true")
    args = parser.parse_args()

    main(args)
