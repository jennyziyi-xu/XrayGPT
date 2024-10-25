import argparse
import os
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn
# import gradio as gr

from xraygpt.common.config import Config
from xraygpt.common.dist_utils import get_rank
from xraygpt.common.registry import registry
from xraygpt.conversation.conversation import Chat, CONV_VISION, Conversation, SeparatorStyle

# imports modules for registration
from xraygpt.datasets.builders import *
from xraygpt.models import *
from xraygpt.processors import *
from xraygpt.runners import *
from xraygpt.tasks import *

def parse_args():
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--cfg-path", required=True, help="path to configuration file.")
    parser.add_argument("--gpu-id", type=int, default=0, help="specify the gpu to load the model.")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )
    args = parser.parse_args()
    return args

print('Initializing Chat')
args = parse_args()
cfg = Config(args)

model_config = cfg.model_cfg
model_config.device_8bit = args.gpu_id
model_cls = registry.get_model_class(model_config.arch)
model = model_cls.from_config(model_config).to('cuda:{}'.format(args.gpu_id))

vis_processor_cfg = cfg.datasets_cfg.openi.vis_processor.train
vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)
chat = Chat(model, vis_processor, device='cuda:{}'.format(args.gpu_id))
print('Initialization Finished')



pre_path = '/n/data1/hms/dbmi/rajpurkar/lab/datasets/cxr/MIMIC-CXR/raw_jpg/files/p10/'

# images = ['p10092355/s59927040/1347b697-d66612bc-acda1c9b-7cd40924-746dbedc.jpg',
#         'p10183375/s50257806/42e0262b-f40d446b-81b71b40-dee28b69-62fdd23b.jpg',
#         'p10277153/s55118290/aaf7b1e2-479c4597-254b57ef-ae936ec2-d0c01ba6.jpg',
#         'p10359261/s51345357/04f42def-b95b54fc-89e95f53-d82d159f-6a9d8ec5.jpg',
#         'p10453519/s52152690/12c002ef-56642f8b-6f1ad349-3da1d7e9-e89e2ba9.jpg']

# i = 0
# for image in images:
image = 'p10453519/s52152690/12c002ef-56642f8b-6f1ad349-3da1d7e9-e89e2ba9.jpg'

image = pre_path + image
img_list = []

CONV_VISION.append_message(CONV_VISION.roles[0], 'Take a look at this chest x-ray and describe the findings and impression.')

chat.upload_img(image, CONV_VISION, img_list)

output_text, _ = chat.answer(CONV_VISION, img_list)
print(output_text)