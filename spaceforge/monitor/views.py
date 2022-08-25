from django.shortcuts import render

import numpy as np
import pandas as pd
from .apps import *
from rest_framework.views import APIView
from rest_framework.response import Response



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

import base64
from io import BytesIO



def encode_image(frame):
    
    return cv2.imencode(".jpg", np.asarray(frame))[1]


class Prediction(APIView):
    
      
         
    def post(self, request):
        variation=False
        #data = request.data
        prompt= request.GET.get('prompt')
        
        print(prompt)
        
        
        height = 512                        # default height
        width = 512                         # default width 
        batch_size = 1


        num_inference_steps = 50            # Number of denoising steps
        guidance_scale = 10                 # Scale for classifier-free guidance

        generator = torch.manual_seed(random.randint(0,25000000))   # Seed generator to create the inital latent noise

        scheduler = LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000)

        text_input = ApiConfig.tokenizer(prompt, padding="max_length", max_length=ApiConfig.tokenizer.model_max_length, truncation=True, return_tensors="pt")

        with torch.no_grad():
          text_embeddings = ApiConfig.text_encoder(text_input.input_ids.to(ApiConfig.torch_device0))[0]

        max_length = text_input.input_ids.shape[-1]
        uncond_input = ApiConfig.tokenizer(
            [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
        )
        with torch.no_grad():
          uncond_embeddings = ApiConfig.text_encoder(uncond_input.input_ids.to(ApiConfig.torch_device0))[0]

        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
        
        latents = torch.randn((batch_size, ApiConfig.unet.in_channels, height // 8, width // 8),
                              generator=generator,
                             )
           
                      
        
        latents = latents.to(ApiConfig.torch_device0)

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
              noise_pred = ApiConfig.unet(latent_model_input, t, encoder_hidden_states=text_embeddings)["sample"]

            # perform guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            # compute the previous noisy sample x_t -> x_t-1
            latents = scheduler.step(noise_pred, i, latents)["prev_sample"]

        # scale and decode the image latents with vae
        latents = 1 / 0.18215 * latents

        with torch.no_grad():
          image = ApiConfig.vae.decode(latents)
                        
                        
        image = (image / 2 + 0.5).clamp(0, 1)
        #img=image.detach().cpu().permute(0, 2,3, 1).squeeze(0).numpy()*255
        image = image.detach().cpu().permute(0, 2, 3, 1).numpy()

        images = (image * 255).round().astype("uint8")
        pil_images = [Image.fromarray(image) for image in images]
        
        
        img_buffer = BytesIO()
        pil_images[0].save(img_buffer, format='JPEG')
        byte_data = img_buffer.getvalue()
        encoded_image = base64.b64encode(byte_data)

      
        #encoded_image=base64.b64encode(pil_images[0])
        
        
        
        #response_dict = {"Output Image": encoded_image}
        
        
        return Response(encoded_image, status=200)