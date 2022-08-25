from django.apps import AppConfig
import os
import joblib
from django.conf import settings


import torch
from torch import autocast, nn
from PIL import Image
import random
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler
from diffusers import LMSDiscreteScheduler
from torch import autocast
import argparse
import cv2
import glob
import os
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer
from realesrgan.archs.srvgg_arch import SRVGGNetCompact

import clip
import os
import numpy as np
import torch.nn.functional as nnf
import sys
from typing import Tuple, List, Union, Optional
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm, trange
import skimage.io as io
import transformers

class MonitorConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'monitor'



class ApiConfig(AppConfig):
    name = 'api'
    #MODEL_FILE = os.path.join(settings.MODELS, "DecisionTreeModel.joblib")
    #model = joblib.load(MODEL_FILE)
    torch_device0 = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_device1 = "cuda:1" if torch.cuda.is_available() else "cpu"
    torch_device2 = "cuda:2" if torch.cuda.is_available() else "cpu"
    torch_device3 = "cuda:3" if torch.cuda.is_available() else "cpu"
    
    path='../../models/'
    vae = AutoencoderKL.from_pretrained(path+'spaceforge-diffusion/vae').to(torch_device0)
    text_encoder = CLIPTextModel.from_pretrained(path+"clip-vit-large-patch14/").to(torch_device0)
    unet = UNet2DConditionModel.from_pretrained(path+"spaceforge-diffusion/unet/").to(torch_device0)
    tokenizer = CLIPTokenizer.from_pretrained(path+"clip-vit-large-patch14/")
    SRmodel = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32,scale=4).to(torch_device1)
    gan_model_path=path+'gans/RealESRGAN_x4plus.pth'
    upsampler = RealESRGANer(scale=4,
                             model_path=gan_model_path,
                             model=SRmodel,
                             tile=0, tile_pad=10,pre_pad=0,
                             half=True)