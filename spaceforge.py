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

class Space:
    def __init__(self):
        
        self.torch_device0 = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.torch_device1 = "cuda:1" if torch.cuda.is_available() else "cpu"
        self.torch_device2 = "cuda:2" if torch.cuda.is_available() else "cpu"
        self.torch_device3 = "cuda:3" if torch.cuda.is_available() else "cpu"

        self.vae = AutoencoderKL.from_pretrained("../models/spaceforge-diffusion/vae").to(self.torch_device0)
        self.text_encoder = CLIPTextModel.from_pretrained("../models/clip-vit-large-patch14/").to(self.torch_device0)
        self.unet = UNet2DConditionModel.from_pretrained("../models/spaceforge-diffusion/unet/").to(self.torch_device0)
        self.tokenizer = CLIPTokenizer.from_pretrained("../models/clip-vit-large-patch14/")
        
        self.SRmodel = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4).to(self.torch_device1)
        self.gan_model_path='../models/gans/RealESRGAN_x4plus.pth'
        self.upsampler = RealESRGANer(scale=4,
                                      model_path=self.gan_model_path,
                                      model=self.SRmodel,
                                      tile=0, tile_pad=10,pre_pad=0,
                                      half=True)
        
        self.stored_latents=[]
        self.id=random.randint(0,25000000)  # also used as random seed
        self.prompt=[]
        
        
    def infer(self, prompt, variation=False):
        print(self.id)
        self.prompt=prompt
                   
        
        height = 512                        # default height
        width = 512                         # default width 
        batch_size = 1


        num_inference_steps = 50            # Number of denoising steps
        guidance_scale = 10                 # Scale for classifier-free guidance

        generator = torch.manual_seed(self.id)   # Seed generator to create the inital latent noise

        scheduler = LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000)

        text_input = self.tokenizer(prompt, padding="max_length", max_length=self.tokenizer.model_max_length, truncation=True, return_tensors="pt")

        with torch.no_grad():
          text_embeddings = self.text_encoder(text_input.input_ids.to(self.torch_device0))[0]

        max_length = text_input.input_ids.shape[-1]
        uncond_input = self.tokenizer(
            [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
        )
        with torch.no_grad():
          uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.torch_device0))[0]

        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
        
        
        if not variation:  #Genrerate a random starting point
            print(variation)
            latents = torch.randn(
              (batch_size, self.unet.in_channels, height // 8, width // 8),
              generator=generator,
            )
            self.stored_latents=latents
        else: 
            print(variation)
            latents=self.varied_latents
                      
        
        latents = latents.to(self.torch_device0)

        scheduler.set_timesteps(num_inference_steps)
        latents = latents * scheduler.sigmas[0]

        with autocast("cuda"):

          for i, t in tqdm(enumerate(scheduler.timesteps)):
            # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
            latent_model_input = torch.cat([latents] * 2)
            sigma = scheduler.sigmas[i]
            latent_model_input = latent_model_input / ((sigma**2 + 1) ** 0.5)

            # predict the noise residual
            with torch.no_grad():
              noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings)["sample"]

            # perform guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            # compute the previous noisy sample x_t -> x_t-1
            latents = scheduler.step(noise_pred, i, latents)["prev_sample"]

        # scale and decode the image latents with vae
        latents = 1 / 0.18215 * latents

        with torch.no_grad():
          image = self.vae.decode(latents)

        image = (image / 2 + 0.5).clamp(0, 1)
        img=image.detach().cpu().permute(0, 2,3, 1).squeeze(0).numpy()*255
        image = image.detach().cpu().permute(0, 2, 3, 1).numpy()

        images = (image * 255).round().astype("uint8")
        pil_images = [Image.fromarray(image) for image in images]
        return pil_images[0]
       
        
    def create_variation(self,vx,vy):
        vx=1.0+vx   
        self.varied_latents=torch.exp((self.stored_latents*vx))+vy
        self.varied_latents=torch.log(self.varied_latents)
        std,mean=torch.std_mean(self.varied_latents)
        self.varied_latents=(self.varied_latents-mean)/std    
        varied_image= self.infer(self.prompt,variation=True)
        
        return varied_image
        
        
        
        