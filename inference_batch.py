import argparse
import os
import sys
import random

import csv

import pandas as pd 

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import gradio as gr

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

sys.path.append(os.path.join(os.path.dirname(__file__), '../UQ/scripts'))
from uncertainty_score import *

input_csv = "/home/jex451/data/mimic_test_reports_new.csv"
pre_path = '/n/data1/hms/dbmi/rajpurkar/lab/datasets/cxr/MIMIC-CXR/raw_jpg/files/'

# TODO: modify these. 
prompt = "Write a detailed radiologic report on the given chest X-ray image. If any support devices are present, describe the support devices."
result_csv_path = "/home/jex451/UQ/outputs/07_04_final/perturbation_diff/inference_3.csv"
result_uq_path = None
logits_csv_path = None

number_samples = 1000
temperature = 0.5

def parse_args():
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--cfg-path",  help="path to configuration file.")
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


if __name__ == "__main__":
    print('Initializing Chat')
    args = parse_args()
    args.cfg_path = "eval_configs/xraygpt_eval.yaml"
    cfg = Config(args)

    model_config = cfg.model_cfg
    model_config.device_8bit = args.gpu_id
    model_cls = registry.get_model_class(model_config.arch)
    model = model_cls.from_config(model_config).to('cuda:{}'.format(args.gpu_id))

    vis_processor_cfg = cfg.datasets_cfg.openi.vis_processor.train
    vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)

    print('Initialization Finished')

    # create a blank new csv file
    # result = pd.DataFrame(columns=['dicom_id', 'study_id', 'subject_id', 'target'])

    # Read from a csv file 
    input_csv = pd.read_csv(input_csv)

    f = open(result_csv_path, 'w')
    f.write("dicom_id,study_id,subject_id,target\n")
    f.flush()
    if result_uq_path:
        f_uq = open(result_uq_path, 'w')
        f_uq.write("data_id,max_prob,max_prob_sentence,max_prob_token,avg_prob,avg_prob_sentence,max_entropy,max_entropy_sentence,max_entropy_token,avg_entropy,avg_entropy_sentence,token_record\n")

    # loop through each row
    for index, row in input_csv.iterrows():

        if (index < number_samples):

            if (index % 20 == 0):
                print("Index is ", index)

            # clear the conversation. 
            CONV_VISION.messages = []
            CONV_VISION.append_message(CONV_VISION.roles[0], prompt)

            chat = Chat(model, vis_processor, device='cuda:{}'.format(args.gpu_id))

            # extract the columns
            dicom_id = row['dicom_id']
            study_id = row['study_id']
            subject_id = row['subject_id']
                
            # construct the image path 
            img_path = pre_path + "p{}/p{}/s{}/{}.jpg".format(str(subject_id)[:2], subject_id, study_id, dicom_id)
            
            img_list = []
            chat.upload_img(img_path, CONV_VISION, img_list)
            output_text, _ = chat.answer(CONV_VISION, img_list, temperature=temperature)

            # write the row to the new csv file
            f.write(f"{dicom_id},{study_id},{subject_id},\"{output_text}\"\n")
            f.flush()

            if result_uq_path:
                uq = get_scores(logits_csv_path)
                f_uq.write(f"{index},{uq[0]},\"{uq[1]}\",\"{uq[2]}\",{uq[3]},\"{uq[4]}\",{uq[5]},\"{uq[6]}\",\"{uq[7]}\",{uq[8]},\"{uq[9]}\",\"{uq[10]}\"\n")


    f.close()
    if result_uq_path:
        f_uq.close()

                
