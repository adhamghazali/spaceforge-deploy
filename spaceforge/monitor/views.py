from django.shortcuts import render

import numpy as np
import pandas as pd
from .apps import *
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import renderers
from ratelimit.decorators import ratelimit

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

import six
from google.cloud import translate_v2 as translate

translate_client = translate.Client()


with open('../data/bad-words.txt') as f:
    list_of_bad_words=f.read().splitlines()
    
warning_image=Image.open('../data/warning.jpg')
    
    
    


def translate_text(target, text):
    """Translates text into the target language.

    Target must be an ISO 639-1 language code.
    See https://g.co/cloud/translate/v2/translate-reference#supported_languages
    """

    if isinstance(text, six.binary_type):
        text = text.decode("utf-8")
    print(text)

    # Text can also be a sequence of strings, in which case this method
    # will return a sequence of results for each text.
    result = translate_client.translate(text, target_language=target)


    return result




def detect_bad_words(prompt,list_of_bad_words):
    words=prompt.split(' ')
    for word in words:
        if word in list_of_bad_words:
            return True
    return False


class JPEGRenderer(renderers.BaseRenderer):
    media_type = 'image/jpeg'
    format = 'jpg'
    charset = None
    render_style = 'binary'

    def render(self, data, media_type=None, renderer_context=None):
        return data



def encode_image(frame):
    
    return cv2.imencode(".jpg", np.asarray(frame))[1]

class P0(APIView):
    
    def post(self, request):
        variation=False
        #data = request.data
        prompt= request.GET.get('prompt')
        result=translate_text('EN',prompt)
        prompt=result['translatedText']   ##English 
        
        if detect_bad_words(prompt,list_of_bad_words):
            out_image = warning_image
            img_buffer = BytesIO()
            out_image.save(img_buffer, format='png')
            byte_data = img_buffer.getvalue()
            encoded_image = base64.b64encode(byte_data)
            return Response(encoded_image,content_type="image/png")
            
        
        
        
        height = 512                        # default height
        width = 512                         # default width 
        batch_size = 1


        num_inference_steps = 200            # Number of denoising steps
        guidance_scale = 10                 # Scale for classifier-free guidance

        generator = torch.manual_seed(random.randint(0,25000000))   # Seed generator to create the inital latent noise

        scheduler = LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000)

        text_input = A0.tokenizer(prompt, padding="max_length", max_length=A0.tokenizer.model_max_length, truncation=True, return_tensors="pt")

        with torch.no_grad():
          text_embeddings = A0.text_encoder(text_input.input_ids.to(A0.torch_device0))[0]

        max_length = text_input.input_ids.shape[-1]
        uncond_input = A0.tokenizer(
            [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
        )
        with torch.no_grad():
          uncond_embeddings = A0.text_encoder(uncond_input.input_ids.to(A0.torch_device0))[0]

        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
        
        latents = torch.randn((batch_size, A0.unet.in_channels, height // 8, width // 8),
                              generator=generator,
                             )
           
                      
        
        latents = latents.to(A0.torch_device0)

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
                noise_pred = A0.unet(latent_model_input, t, encoder_hidden_states=text_embeddings)["sample"]

            # perform guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            # compute the previous noisy sample x_t -> x_t-1
            latents = scheduler.step(noise_pred, i, latents)["prev_sample"]

        # scale and decode the image latents with vae
        latents = 1 / 0.18215 * latents

        with torch.no_grad():
          image = A0.vae.decode(latents)
        
                        
                        
        image = (image / 2 + 0.5).clamp(0, 1)
        img=image.detach().cpu().permute(0, 2,3, 1).squeeze(0).numpy()*255
        image = image.detach().cpu().permute(0, 2, 3, 1).numpy()

        images = (image * 255).round().astype("uint8")
        pil_images = [Image.fromarray(image) for image in images]
        
        output, _ = A0.upsampler.enhance(img, outscale=4)
        out_image = Image.fromarray(output)
        
        
        
        img_buffer = BytesIO()
        out_image.save(img_buffer, format='JPEG')
        byte_data = img_buffer.getvalue()
        encoded_image = base64.b64encode(byte_data)
        
        #image_data = base64.b64encode(encoded_image).decode('utf-8')
        
        #renderer=JPEGRenderer()
    
        
        
        return Response(encoded_image,content_type="image/png")
    
    def get(self, request):
        
        print('this is a get request')
        variation=False
        #data = request.data
        prompt= request.GET.get('prompt')
        print(prompt)
        result=translate_text('EN',prompt)
        prompt=result['translatedText']
        
        if detect_bad_words(prompt,list_of_bad_words):
            out_image = warning_image
            img_buffer = BytesIO()
            out_image.save(img_buffer, format='png')
            byte_data = img_buffer.getvalue()
            encoded_image = base64.b64encode(byte_data)
            image_data = base64.b64encode(byte_data).decode('utf-8')
            ctx={'image':image_data}
            
            return  render(request, 'index.html',ctx)
        


                
        
        height = 512                        # default height
        width = 512                         # default width 
        batch_size = 1


        num_inference_steps = 200            # Number of denoising steps
        guidance_scale = 10                 # Scale for classifier-free guidance

        generator = torch.manual_seed(random.randint(0,25000000))   # Seed generator to create the inital latent noise

        scheduler = LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000)

        text_input = A0.tokenizer(prompt, padding="max_length", max_length=A0.tokenizer.model_max_length, truncation=True, return_tensors="pt")

        with torch.no_grad():
          text_embeddings = A0.text_encoder(text_input.input_ids.to(A0.torch_device0))[0]

        max_length = text_input.input_ids.shape[-1]
        uncond_input = A0.tokenizer(
            [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
        )
        with torch.no_grad():
          uncond_embeddings = A0.text_encoder(uncond_input.input_ids.to(A0.torch_device0))[0]

        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
        
        latents = torch.randn((batch_size, A0.unet.in_channels, height // 8, width // 8),
                              generator=generator,
                             )
           
                      
        
        latents = latents.to(A0.torch_device0)

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
              noise_pred = A0.unet(latent_model_input, t, encoder_hidden_states=text_embeddings)["sample"]

            # perform guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            # compute the previous noisy sample x_t -> x_t-1
            latents = scheduler.step(noise_pred, i, latents)["prev_sample"]

        # scale and decode the image latents with vae
        latents = 1 / 0.18215 * latents

        with torch.no_grad():
          image = A0.vae.decode(latents)
                        
                        
        image = (image / 2 + 0.5).clamp(0, 1)
        img=image.detach().cpu().permute(0, 2,3, 1).squeeze(0).numpy()*255
        image = image.detach().cpu().permute(0, 2, 3, 1).numpy()

        images = (image * 255).round().astype("uint8")
        pil_images = [Image.fromarray(image) for image in images]
        
        output, _ = A0.upsampler.enhance(img, outscale=2)
        out_image = Image.fromarray(output)
        
        
        
        img_buffer = BytesIO()
        out_image.save(img_buffer, format='png')
        byte_data = img_buffer.getvalue()
        encoded_image = base64.b64encode(byte_data)
        
        image_data = base64.b64encode(byte_data).decode('utf-8')
        
        
        ctx={'image':image_data}
        
        #ctx["image"] = image_data
        
        return  render(request, 'index.html',ctx)
        
        
    
    
class P1(APIView):
    
    def post(self, request):
        variation=False
        #data = request.data
        prompt= request.GET.get('prompt')
        result=translate_text('EN',prompt)
        prompt=result['translatedText']   ##English 
        
        if detect_bad_words(prompt,list_of_bad_words):
            out_image = warning_image
            img_buffer = BytesIO()
            out_image.save(img_buffer, format='png')
            byte_data = img_buffer.getvalue()
            encoded_image = base64.b64encode(byte_data)
            return Response(encoded_image,content_type="image/png")
            
        
        
        
        height = 512                        # default height
        width = 512                         # default width 
        batch_size = 1


        num_inference_steps = 200            # Number of denoising steps
        guidance_scale = 10                 # Scale for classifier-free guidance

        generator = torch.manual_seed(random.randint(0,25000000))   # Seed generator to create the inital latent noise

        scheduler = LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000)

        text_input = A1.tokenizer(prompt, padding="max_length", max_length=A1.tokenizer.model_max_length, truncation=True, return_tensors="pt")

        with torch.no_grad():
          text_embeddings = A1.text_encoder(text_input.input_ids.to(A1.torch_device1))[0]

        max_length = text_input.input_ids.shape[-1]
        uncond_input = A1.tokenizer(
            [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
        )
        with torch.no_grad():
          uncond_embeddings = A1.text_encoder(uncond_input.input_ids.to(A1.torch_device1))[0]

        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
        
        latents = torch.randn((batch_size, A1.unet.in_channels, height // 8, width // 8),
                              generator=generator,
                             )
           
                      
        
        latents = latents.to(A1.torch_device1)

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
                noise_pred = A1.unet(latent_model_input, t, encoder_hidden_states=text_embeddings)["sample"]

            # perform guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            # compute the previous noisy sample x_t -> x_t-1
            latents = scheduler.step(noise_pred, i, latents)["prev_sample"]

        # scale and decode the image latents with vae
        latents = 1 / 0.18215 * latents

        with torch.no_grad():
          image = A1.vae.decode(latents)
        
                        
                        
        image = (image / 2 + 0.5).clamp(0, 1)
        img=image.detach().cpu().permute(0, 2,3, 1).squeeze(0).numpy()*255
        image = image.detach().cpu().permute(0, 2, 3, 1).numpy()

        images = (image * 255).round().astype("uint8")
        pil_images = [Image.fromarray(image) for image in images]
        
        output, _ = A1.upsampler.enhance(img, outscale=4)
        out_image = Image.fromarray(output)
        
        
        
        img_buffer = BytesIO()
        out_image.save(img_buffer, format='JPEG')
        byte_data = img_buffer.getvalue()
        encoded_image = base64.b64encode(byte_data)
        
        #image_data = base64.b64encode(encoded_image).decode('utf-8')
        
        #renderer=JPEGRenderer()
    
        
        
        return Response(encoded_image,content_type="image/png")
    
    def get(self, request):
        
        print('this is a get request')
        variation=False
        #data = request.data
        prompt= request.GET.get('prompt')
        print(prompt)
        result=translate_text('EN',prompt)
        prompt=result['translatedText']
        
        if detect_bad_words(prompt,list_of_bad_words):
            out_image = warning_image
            img_buffer = BytesIO()
            out_image.save(img_buffer, format='png')
            byte_data = img_buffer.getvalue()
            encoded_image = base64.b64encode(byte_data)
            image_data = base64.b64encode(byte_data).decode('utf-8')
            ctx={'image':image_data}
            
            return  render(request, 'index.html',ctx)
        


                
        
        height = 512                        # default height
        width = 512                         # default width 
        batch_size = 1


        num_inference_steps = 200            # Number of denoising steps
        guidance_scale = 10                 # Scale for classifier-free guidance

        generator = torch.manual_seed(random.randint(0,25000000))   # Seed generator to create the inital latent noise

        scheduler = LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000)

        text_input = A1.tokenizer(prompt, padding="max_length", max_length=A1.tokenizer.model_max_length, truncation=True, return_tensors="pt")

        with torch.no_grad():
          text_embeddings = A1.text_encoder(text_input.input_ids.to(A1.torch_device1))[0]

        max_length = text_input.input_ids.shape[-1]
        uncond_input = A1.tokenizer(
            [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
        )
        with torch.no_grad():
          uncond_embeddings = A1.text_encoder(uncond_input.input_ids.to(A1.torch_device1))[0]

        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
        
        latents = torch.randn((batch_size, A1.unet.in_channels, height // 8, width // 8),
                              generator=generator,
                             )
           
                      
        
        latents = latents.to(A1.torch_device1)

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
              noise_pred = A1.unet(latent_model_input, t, encoder_hidden_states=text_embeddings)["sample"]

            # perform guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            # compute the previous noisy sample x_t -> x_t-1
            latents = scheduler.step(noise_pred, i, latents)["prev_sample"]

        # scale and decode the image latents with vae
        latents = 1 / 0.18215 * latents

        with torch.no_grad():
          image = A1.vae.decode(latents)
                        
                        
        image = (image / 2 + 0.5).clamp(0, 1)
        img=image.detach().cpu().permute(0, 2,3, 1).squeeze(0).numpy()*255
        image = image.detach().cpu().permute(0, 2, 3, 1).numpy()

        images = (image * 255).round().astype("uint8")
        pil_images = [Image.fromarray(image) for image in images]
        
        output, _ = A1.upsampler.enhance(img, outscale=2)
        out_image = Image.fromarray(output)
        
        
        
        img_buffer = BytesIO()
        out_image.save(img_buffer, format='png')
        byte_data = img_buffer.getvalue()
        encoded_image = base64.b64encode(byte_data)
        
        image_data = base64.b64encode(byte_data).decode('utf-8')
        
        
        ctx={'image':image_data}
        
        #ctx["image"] = image_data
        
        return  render(request, 'index.html',ctx)
    
    
class P2(APIView):
    
    def post(self, request):
        variation=False
        #data = request.data
        prompt= request.GET.get('prompt')
        result=translate_text('EN',prompt)
        prompt=result['translatedText']   ##English 
        
        if detect_bad_words(prompt,list_of_bad_words):
            out_image = warning_image
            img_buffer = BytesIO()
            out_image.save(img_buffer, format='png')
            byte_data = img_buffer.getvalue()
            encoded_image = base64.b64encode(byte_data)
            return Response(encoded_image,content_type="image/png")
            
        
        
        
        height = 512                        # default height
        width = 512                         # default width 
        batch_size = 1


        num_inference_steps = 200            # Number of denoising steps
        guidance_scale = 10                 # Scale for classifier-free guidance

        generator = torch.manual_seed(random.randint(0,25000000))   # Seed generator to create the inital latent noise

        scheduler = LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000)

        text_input = A2.tokenizer(prompt, padding="max_length", max_length=A2.tokenizer.model_max_length, truncation=True, return_tensors="pt")

        with torch.no_grad():
          text_embeddings = A2.text_encoder(text_input.input_ids.to(A2.torch_device2))[0]

        max_length = text_input.input_ids.shape[-1]
        uncond_input = A2.tokenizer(
            [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
        )
        with torch.no_grad():
          uncond_embeddings = A2.text_encoder(uncond_input.input_ids.to(A2.torch_device2))[0]

        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
        
        latents = torch.randn((batch_size, A2.unet.in_channels, height // 8, width // 8),
                              generator=generator,
                             )
           
                      
        
        latents = latents.to(A2.torch_device2)

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
                noise_pred = A2.unet(latent_model_input, t, encoder_hidden_states=text_embeddings)["sample"]

            # perform guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            # compute the previous noisy sample x_t -> x_t-1
            latents = scheduler.step(noise_pred, i, latents)["prev_sample"]

        # scale and decode the image latents with vae
        latents = 1 / 0.18215 * latents

        with torch.no_grad():
          image = A2.vae.decode(latents)
        
                        
                        
        image = (image / 2 + 0.5).clamp(0, 1)
        img=image.detach().cpu().permute(0, 2,3, 1).squeeze(0).numpy()*255
        image = image.detach().cpu().permute(0, 2, 3, 1).numpy()

        images = (image * 255).round().astype("uint8")
        pil_images = [Image.fromarray(image) for image in images]
        
        output, _ = A2.upsampler.enhance(img, outscale=4)
        out_image = Image.fromarray(output)
        
        
        
        img_buffer = BytesIO()
        out_image.save(img_buffer, format='JPEG')
        byte_data = img_buffer.getvalue()
        encoded_image = base64.b64encode(byte_data)
        
        #image_data = base64.b64encode(encoded_image).decode('utf-8')
        
        #renderer=JPEGRenderer()
    
        
        
        return Response(encoded_image,content_type="image/png")
    
    def get(self, request):
        
        print('this is a get request')
        variation=False
        #data = request.data
        prompt= request.GET.get('prompt')
        print(prompt)
        result=translate_text('EN',prompt)
        prompt=result['translatedText']
        
        if detect_bad_words(prompt,list_of_bad_words):
            out_image = warning_image
            img_buffer = BytesIO()
            out_image.save(img_buffer, format='png')
            byte_data = img_buffer.getvalue()
            encoded_image = base64.b64encode(byte_data)
            image_data = base64.b64encode(byte_data).decode('utf-8')
            ctx={'image':image_data}
            
            return  render(request, 'index.html',ctx)
        


                
        
        height = 512                        # default height
        width = 512                         # default width 
        batch_size = 1


        num_inference_steps = 200            # Number of denoising steps
        guidance_scale = 10                 # Scale for classifier-free guidance

        generator = torch.manual_seed(random.randint(0,25000000))   # Seed generator to create the inital latent noise

        scheduler = LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000)

        text_input = A2.tokenizer(prompt, padding="max_length", max_length=A2.tokenizer.model_max_length, truncation=True, return_tensors="pt")

        with torch.no_grad():
          text_embeddings = A2.text_encoder(text_input.input_ids.to(A2.torch_device2))[0]

        max_length = text_input.input_ids.shape[-1]
        uncond_input = A2.tokenizer(
            [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
        )
        with torch.no_grad():
          uncond_embeddings = A2.text_encoder(uncond_input.input_ids.to(A2.torch_device2))[0]

        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
        
        latents = torch.randn((batch_size, A2.unet.in_channels, height // 8, width // 8),
                              generator=generator,
                             )
           
                      
        
        latents = latents.to(A2.torch_device2)

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
              noise_pred = A2.unet(latent_model_input, t, encoder_hidden_states=text_embeddings)["sample"]

            # perform guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            # compute the previous noisy sample x_t -> x_t-1
            latents = scheduler.step(noise_pred, i, latents)["prev_sample"]

        # scale and decode the image latents with vae
        latents = 1 / 0.18215 * latents

        with torch.no_grad():
          image = A2.vae.decode(latents)
                        
                        
        image = (image / 2 + 0.5).clamp(0, 1)
        img=image.detach().cpu().permute(0, 2,3, 1).squeeze(0).numpy()*255
        image = image.detach().cpu().permute(0, 2, 3, 1).numpy()

        images = (image * 255).round().astype("uint8")
        pil_images = [Image.fromarray(image) for image in images]
        
        output, _ = A2.upsampler.enhance(img, outscale=2)
        out_image = Image.fromarray(output)
        
        
        
        img_buffer = BytesIO()
        out_image.save(img_buffer, format='png')
        byte_data = img_buffer.getvalue()
        encoded_image = base64.b64encode(byte_data)
        
        image_data = base64.b64encode(byte_data).decode('utf-8')
        
        
        ctx={'image':image_data}
        
        #ctx["image"] = image_data
        
        return  render(request, 'index.html',ctx)
        
        
    
    
    
class P3(APIView):
    
    def post(self, request):
        variation=False
        #data = request.data
        prompt= request.GET.get('prompt')
        result=translate_text('EN',prompt)
        prompt=result['translatedText']   ##English 
        
        if detect_bad_words(prompt,list_of_bad_words):
            out_image = warning_image
            img_buffer = BytesIO()
            out_image.save(img_buffer, format='png')
            byte_data = img_buffer.getvalue()
            encoded_image = base64.b64encode(byte_data)
            return Response(encoded_image,content_type="image/png")
            
        
        
        
        height = 512                        # default height
        width = 512                         # default width 
        batch_size = 1


        num_inference_steps = 200            # Number of denoising steps
        guidance_scale = 10                 # Scale for classifier-free guidance

        generator = torch.manual_seed(random.randint(0,25000000))   # Seed generator to create the inital latent noise

        scheduler = LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000)

        text_input = A3.tokenizer(prompt, padding="max_length", max_length=A3.tokenizer.model_max_length, truncation=True, return_tensors="pt")

        with torch.no_grad():
          text_embeddings = A3.text_encoder(text_input.input_ids.to(A3.torch_device3))[0]

        max_length = text_input.input_ids.shape[-1]
        uncond_input = A2.tokenizer(
            [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
        )
        with torch.no_grad():
          uncond_embeddings = A3.text_encoder(uncond_input.input_ids.to(A3.torch_device3))[0]

        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
        
        latents = torch.randn((batch_size, A3.unet.in_channels, height // 8, width // 8),
                              generator=generator,
                             )
           
                      
        
        latents = latents.to(A3.torch_device3)

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
                noise_pred = A3.unet(latent_model_input, t, encoder_hidden_states=text_embeddings)["sample"]

            # perform guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            # compute the previous noisy sample x_t -> x_t-1
            latents = scheduler.step(noise_pred, i, latents)["prev_sample"]

        # scale and decode the image latents with vae
        latents = 1 / 0.18215 * latents

        with torch.no_grad():
          image = A3.vae.decode(latents)
        
                        
                        
        image = (image / 2 + 0.5).clamp(0, 1)
        img=image.detach().cpu().permute(0, 2,3, 1).squeeze(0).numpy()*255
        image = image.detach().cpu().permute(0, 2, 3, 1).numpy()

        images = (image * 255).round().astype("uint8")
        pil_images = [Image.fromarray(image) for image in images]
        
        output, _ = A3.upsampler.enhance(img, outscale=4)
        out_image = Image.fromarray(output)
        
        
        
        img_buffer = BytesIO()
        out_image.save(img_buffer, format='JPEG')
        byte_data = img_buffer.getvalue()
        encoded_image = base64.b64encode(byte_data)
        
        #image_data = base64.b64encode(encoded_image).decode('utf-8')
        
        #renderer=JPEGRenderer()
    
        
        
        return Response(encoded_image,content_type="image/png")
    
    def get(self, request):
        
        print('this is a get request')
        variation=False
        #data = request.data
        prompt= request.GET.get('prompt')
        print(prompt)
        result=translate_text('EN',prompt)
        prompt=result['translatedText']
        
        if detect_bad_words(prompt,list_of_bad_words):
            out_image = warning_image
            img_buffer = BytesIO()
            out_image.save(img_buffer, format='png')
            byte_data = img_buffer.getvalue()
            encoded_image = base64.b64encode(byte_data)
            image_data = base64.b64encode(byte_data).decode('utf-8')
            ctx={'image':image_data}
            
            return  render(request, 'index.html',ctx)
        


                
        
        height = 512                        # default height
        width = 512                         # default width 
        batch_size = 1


        num_inference_steps = 200            # Number of denoising steps
        guidance_scale = 10                 # Scale for classifier-free guidance

        generator = torch.manual_seed(random.randint(0,25000000))   # Seed generator to create the inital latent noise

        scheduler = LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000)

        text_input = A3.tokenizer(prompt, padding="max_length", max_length=A3.tokenizer.model_max_length, truncation=True, return_tensors="pt")

        with torch.no_grad():
          text_embeddings = A3.text_encoder(text_input.input_ids.to(A3.torch_device3))[0]

        max_length = text_input.input_ids.shape[-1]
        uncond_input = A3.tokenizer(
            [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
        )
        with torch.no_grad():
          uncond_embeddings = A3.text_encoder(uncond_input.input_ids.to(A3.torch_device3))[0]

        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
        
        latents = torch.randn((batch_size, A2.unet.in_channels, height // 8, width // 8),
                              generator=generator,
                             )
           
                      
        
        latents = latents.to(A3.torch_device3)

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
              noise_pred = A3.unet(latent_model_input, t, encoder_hidden_states=text_embeddings)["sample"]

            # perform guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            # compute the previous noisy sample x_t -> x_t-1
            latents = scheduler.step(noise_pred, i, latents)["prev_sample"]

        # scale and decode the image latents with vae
        latents = 1 / 0.18215 * latents

        with torch.no_grad():
          image = A3.vae.decode(latents)
                        
                        
        image = (image / 2 + 0.5).clamp(0, 1)
        img=image.detach().cpu().permute(0, 2,3, 1).squeeze(0).numpy()*255
        image = image.detach().cpu().permute(0, 2, 3, 1).numpy()

        images = (image * 255).round().astype("uint8")
        pil_images = [Image.fromarray(image) for image in images]
        
        output, _ = A3.upsampler.enhance(img, outscale=2)
        out_image = Image.fromarray(output)
        
        
        
        img_buffer = BytesIO()
        out_image.save(img_buffer, format='png')
        byte_data = img_buffer.getvalue()
        encoded_image = base64.b64encode(byte_data)
        
        image_data = base64.b64encode(byte_data).decode('utf-8')
        
        
        ctx={'image':image_data}
        
        #ctx["image"] = image_data
        
        return  render(request, 'index.html',ctx)
        
        
    
    
    
class P4(APIView):
    
    def post(self, request):
        variation=False
        #data = request.data
        prompt= request.GET.get('prompt')
        result=translate_text('EN',prompt)
        prompt=result['translatedText']   ##English 
        
        if detect_bad_words(prompt,list_of_bad_words):
            out_image = warning_image
            img_buffer = BytesIO()
            out_image.save(img_buffer, format='png')
            byte_data = img_buffer.getvalue()
            encoded_image = base64.b64encode(byte_data)
            return Response(encoded_image,content_type="image/png")
            
        
        
        
        height = 512                        # default height
        width = 512                         # default width 
        batch_size = 1


        num_inference_steps = 200            # Number of denoising steps
        guidance_scale = 10                 # Scale for classifier-free guidance

        generator = torch.manual_seed(random.randint(0,25000000))   # Seed generator to create the inital latent noise

        scheduler = LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000)

        text_input = A4.tokenizer(prompt, padding="max_length", max_length=A4.tokenizer.model_max_length, truncation=True, return_tensors="pt")

        with torch.no_grad():
          text_embeddings = A4.text_encoder(text_input.input_ids.to(A4.torch_device4))[0]

        max_length = text_input.input_ids.shape[-1]
        uncond_input = A4.tokenizer(
            [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
        )
        with torch.no_grad():
          uncond_embeddings = A4.text_encoder(uncond_input.input_ids.to(A4.torch_device4))[0]

        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
        
        latents = torch.randn((batch_size, A4.unet.in_channels, height // 8, width // 8),
                              generator=generator,
                             )
           
                      
        
        latents = latents.to(A4.torch_device4)

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
                noise_pred = A4.unet(latent_model_input, t, encoder_hidden_states=text_embeddings)["sample"]

            # perform guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            # compute the previous noisy sample x_t -> x_t-1
            latents = scheduler.step(noise_pred, i, latents)["prev_sample"]

        # scale and decode the image latents with vae
        latents = 1 / 0.18215 * latents

        with torch.no_grad():
          image = A4.vae.decode(latents)
        
                        
                        
        image = (image / 2 + 0.5).clamp(0, 1)
        img=image.detach().cpu().permute(0, 2,3, 1).squeeze(0).numpy()*255
        image = image.detach().cpu().permute(0, 2, 3, 1).numpy()

        images = (image * 255).round().astype("uint8")
        pil_images = [Image.fromarray(image) for image in images]
        
        output, _ = A4.upsampler.enhance(img, outscale=4)
        out_image = Image.fromarray(output)
        
        
        
        img_buffer = BytesIO()
        out_image.save(img_buffer, format='JPEG')
        byte_data = img_buffer.getvalue()
        encoded_image = base64.b64encode(byte_data)
        
        #image_data = base64.b64encode(encoded_image).decode('utf-8')
        
        #renderer=JPEGRenderer()
    
        
        
        return Response(encoded_image,content_type="image/png")
    
    def get(self, request):
        
        print('this is a get request')
        variation=False
        #data = request.data
        prompt= request.GET.get('prompt')
        print(prompt)
        result=translate_text('EN',prompt)
        prompt=result['translatedText']
        
        if detect_bad_words(prompt,list_of_bad_words):
            out_image = warning_image
            img_buffer = BytesIO()
            out_image.save(img_buffer, format='png')
            byte_data = img_buffer.getvalue()
            encoded_image = base64.b64encode(byte_data)
            image_data = base64.b64encode(byte_data).decode('utf-8')
            ctx={'image':image_data}
            
            return  render(request, 'index.html',ctx)
        


                
        
        height = 512                        # default height
        width = 512                         # default width 
        batch_size = 1


        num_inference_steps = 200            # Number of denoising steps
        guidance_scale = 10                 # Scale for classifier-free guidance

        generator = torch.manual_seed(random.randint(0,25000000))   # Seed generator to create the inital latent noise

        scheduler = LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000)

        text_input = A4.tokenizer(prompt, padding="max_length", max_length=A4.tokenizer.model_max_length, truncation=True, return_tensors="pt")

        with torch.no_grad():
          text_embeddings = A4.text_encoder(text_input.input_ids.to(A4.torch_device4))[0]

        max_length = text_input.input_ids.shape[-1]
        uncond_input = A4.tokenizer(
            [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
        )
        with torch.no_grad():
          uncond_embeddings = A4.text_encoder(uncond_input.input_ids.to(A4.torch_device4))[0]

        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
        
        latents = torch.randn((batch_size, A4.unet.in_channels, height // 8, width // 8),
                              generator=generator,
                             )
           
                      
        
        latents = latents.to(A4.torch_device4)

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
              noise_pred = A4.unet(latent_model_input, t, encoder_hidden_states=text_embeddings)["sample"]

            # perform guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            # compute the previous noisy sample x_t -> x_t-1
            latents = scheduler.step(noise_pred, i, latents)["prev_sample"]

        # scale and decode the image latents with vae
        latents = 1 / 0.18215 * latents

        with torch.no_grad():
          image = A4.vae.decode(latents)
                        
                        
        image = (image / 2 + 0.5).clamp(0, 1)
        img=image.detach().cpu().permute(0, 2,3, 1).squeeze(0).numpy()*255
        image = image.detach().cpu().permute(0, 2, 3, 1).numpy()

        images = (image * 255).round().astype("uint8")
        pil_images = [Image.fromarray(image) for image in images]
        
        output, _ = A4.upsampler.enhance(img, outscale=2)
        out_image = Image.fromarray(output)
        
        
        
        img_buffer = BytesIO()
        out_image.save(img_buffer, format='png')
        byte_data = img_buffer.getvalue()
        encoded_image = base64.b64encode(byte_data)
        
        image_data = base64.b64encode(byte_data).decode('utf-8')
        
        
        ctx={'image':image_data}
        
        #ctx["image"] = image_data
        
        return  render(request, 'index.html',ctx)
        
        