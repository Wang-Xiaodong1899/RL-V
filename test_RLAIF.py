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

# model_path = "/mnt/storage/user/wangxiaodong/RLAIF-V/RLAIF-V-7B"
# model_name='llava-v1.5-7b'
# tokenizer, model, image_processor, context_len = load_pretrained_model(
# model_path, model_base=None,model_name=model_name, device_map={"": 'cuda'})

