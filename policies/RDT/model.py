#!/home/lin/software/miniconda3/envs/aloha/bin/python
# -- coding: UTF-8
"""
#!/usr/bin/python3
"""
from pathlib import Path

# get current workspace
current_file = Path(__file__)

import json
import sys
parent_dir = current_file.parent
sys.path.append(str(parent_dir))

import os

import argparse

import threading
import time
import yaml
from collections import deque

import numpy as np
import torch
from PIL import Image as PImage
import cv2

from scripts.agilex_model import create_model
from models.multimodal_encoder.t5_encoder import T5Embedder
global_path = parent_dir.parent

class RDT:
    def __init__(self, pretrained_model_name_or_path,task_name):
        # set path
        current_file = Path(__file__)
        self.global_path = current_file.parent
        # load the config
        self.config = {
            'episode_len': 10000, # args.max_publish_step
            'state_dim': 7, #todo # 14 dims action:[left joint angles,left gripper,right joint angles,right gripper]
            'chunk_size': 64, #args.chunk_size
            'camera_names': ['cam_high', 'cam_right_wrist', 'cam_left_wrist'],
        }
        # setup config
        self.args = {
            'max_publish_step': 10000,  # Maximum number of action publishing steps
            'seed': None,              # Random seed
            'ctrl_freq': 25,           # The control frequency of the robot
            'chunk_size': 64,          # Action chunk size
            # 'disable_puppet_arm': False,  # Whether to disable the puppet arm
            'config_path': os.path.join(self.global_path,"configs/base.yaml"), 
            "pretrained_model_name_or_path": pretrained_model_name_or_path
        }
        
        # Load rdt model
        self.policy = self.make_policy(self.args)
        self.max_publish_step = self.config['episode_len']
        self.chunk_size = self.config['chunk_size']
        self.task_name =task_name
        self.observation_window = None   
        self.img_size = (640,480)
        # self.set_language_embed() # 不加载语言模型，使用预先处理的lang_embed
        
    # set img_size
    def set_img_size(self,img_size):
        self.img_size = img_size

    def set_language_embed(self):
        GPU = 0
        MODEL_PATH = os.path.join(self.global_path,"weights/RDT/t5-v1_1-xxl")
        CONFIG_PATH = os.path.join(self.global_path,"configs/base.yaml")
        with open(CONFIG_PATH, "r") as fp:
            config = yaml.safe_load(fp)
        device = torch.device(f"cuda:{GPU}")
        text_embedder = T5Embedder(
            from_pretrained=MODEL_PATH, 
            model_max_length=config["dataset"]["tokenizer_max_length"], 
            device=device,
            use_offload_folder=None
        )
        self.tokenizer, self.text_encoder = text_embedder.tokenizer, text_embedder.model
    
    # set language randomly
    def random_set_language(self):
        json_Path =f"training_data/instructions/{self.task_name}.json"
        with open(json_Path, 'r') as f_instr:
            instruction_dict = json.load(f_instr)
        instructions = instruction_dict['instructions']
        index = np.random.randint(0, len(instructions)) 
        # instruction = np.random.choice(instructions)
        self.set_language_instruction(index,instructions[index])
        
    # encoding language 
    def set_language_instruction(self, index, language_instruction, save_dir=None,task_name=None):
        assert ((save_dir is None) ^ (task_name is None)) == False, "input error"
        lang_embed_path = f"training_data/{self.task_name}/instructions/lang_embed_{index}.pt"
        if os.path.isfile(lang_embed_path):
            self.lang_embeddings = torch.load(lang_embed_path).unsqueeze(0)
            print("loading instruction from pre-embed path")
        else:
            tokens = self.tokenizer(
                language_instruction, return_tensors="pt",
                padding="longest",
                truncation=True
            )["input_ids"].to(0)

            tokens = tokens.view(1, -1)
            with torch.no_grad():
                pred = self.text_encoder(tokens).last_hidden_state.detach().cpu()
            if save_dir != None:
                save_path = os.path.join(save_dir,f"{task_name}.pt")
                torch.save({
                    "name": task_name,
                    "instruction": language_instruction,
                    "embeddings": pred
                    }, save_path
                )
            self.lang_embeddings = pred
            print("self.lang_embeddings:",self.lang_embeddings.shape) #self.lang_embeddings: torch.Size([1, 17, 4096])
        print(f"successfully set instruction:{language_instruction}")
    
    # Update the observation window buffer
    def update_observation_window(self, img_arr, state):
        # JPEG transformation
        # Align with training
        def jpeg_mapping(img):
            if img is None:
                return None
            img = cv2.imencode('.jpg', img)[1].tobytes()
            img = cv2.imdecode(np.frombuffer(img, np.uint8), cv2.IMREAD_COLOR)
            return img
        def resize_img(img,size):
            return cv2.resize(img, size)
        if self.observation_window is None:
            self.observation_window = deque(maxlen=2)

            # Append the first dummy image
            self.observation_window.append(
                {
                    'qpos': None,
                    'images':
                        {
                            self.config["camera_names"][0]: None,
                            self.config["camera_names"][1]: None,
                            self.config["camera_names"][2]: None,
                        },
                }
            )

        img_front, img_right, img_left, puppet_arm = img_arr[0], img_arr[1], img_arr[2], state
        # 如果是 (C, H, W)，转为 (H, W, C)
        if img_front.ndim == 3 and img_front.shape[0] in [1, 3, 4]:
            img_front = np.transpose(img_front, (1, 2, 0))
        if img_right.ndim == 3 and img_right.shape[0] in [1, 3, 4]:
            img_right = np.transpose(img_right, (1, 2, 0))
        if img_left.ndim == 3 and img_left.shape[0] in [1, 3, 4]:
            img_left = np.transpose(img_left, (1, 2, 0))
        # img resize
        img_front = resize_img(img_front,self.img_size)
        img_left = resize_img(img_left,self.img_size)
        img_right = resize_img(img_right,self.img_size)
        # img jprg encoding
        img_front = jpeg_mapping(img_front)
        img_left = jpeg_mapping(img_left)
        img_right = jpeg_mapping(img_right)

        qpos = np.array(puppet_arm)
        qpos = torch.from_numpy(qpos).float().cuda()
        self.observation_window.append(
            {
                'qpos': qpos,
                'images':
                    {
                        self.config["camera_names"][0]: img_front,
                        self.config["camera_names"][1]: img_right,
                        self.config["camera_names"][2]: img_left,
                    },
            }
        )

    def get_action(self, img_arr=None, state=None):
        assert (img_arr is None) ^ (state is None) == False, "input error"
        if (img_arr is not None) and (state is not None):
            self.update_observation_window(img_arr, state)

        with torch.inference_mode():
            action_buffer = inference_fn(self.config, self.policy, self.lang_embeddings,self.observation_window).copy()

        return action_buffer

    def reset_obsrvationwindows(self):
        self.lang_embeddings = None
        self.observation_window = None
        print("successfully unset obs and language intruction")

    # Initialize the model
    def make_policy(self,args):
        with open(args["config_path"], "r") as fp:
            config_base_yaml = yaml.safe_load(fp)
        args["config"] = config_base_yaml
        # pretrained_text_encoder_name_or_path = "weights/RDT/t5-v1_1-xxl"
        pretrained_vision_encoder_name_or_path = os.path.join(self.global_path,"weights/RDT/siglip-so400m-patch14-384")
        model = create_model(
            args=args["config"],
            dtype=torch.bfloat16,
            pretrained=args["pretrained_model_name_or_path"],
            # pretrained_text_encoder_name_or_path=pretrained_text_encoder_name_or_path,
            pretrained_vision_encoder_name_or_path=pretrained_vision_encoder_name_or_path,
            control_frequency=args["ctrl_freq"],
        )

        return model

# RDT inference
def inference_fn(config, policy,lang_embeddings,observation_window):

    # print(f"Start inference_thread_fn: t={t}")
    while True:
        time1 = time.time()

        # fetch images in sequence [front, right, left]
        image_arrs = [
            observation_window[-2]['images'][config['camera_names'][0]],
            observation_window[-2]['images'][config['camera_names'][1]],
            observation_window[-2]['images'][config['camera_names'][2]],

            observation_window[-1]['images'][config['camera_names'][0]],
            observation_window[-1]['images'][config['camera_names'][1]],
            observation_window[-1]['images'][config['camera_names'][2]]
        ]

        images = [PImage.fromarray(arr) if arr is not None else None
                  for arr in image_arrs]

        # get last qpos in shape [14, ]
        proprio = observation_window[-1]['qpos']
        # unsqueeze to [1, 14]
        proprio = proprio.unsqueeze(0)

        # actions shaped as [1, 64, 14] in format [left, right]
        actions = policy.step(
            proprio=proprio,
            images=images,
            text_embeds=lang_embeddings
        ).squeeze(0).cpu().numpy()
        # print(f"inference_actions: {actions.squeeze()}")

        # print(f"Model inference time: {time.time() - time1} s")

        # print(f"Finish inference_thread_fn: t={t}")
        return actions
