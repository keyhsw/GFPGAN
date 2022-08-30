import os

import cv2
import gradio as gr
import torch
from basicsr.archs.srvgg_arch import SRVGGNetCompact
from gfpgan.utils import GFPGANer
from realesrgan.utils import RealESRGANer

os.system("pip freeze")
os.system(
    "wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth -P .")
os.system("wget https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.2.pth -P .")
os.system("wget https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth -P .")

torch.hub.download_url_to_file(
    'https://upload.wikimedia.org/wikipedia/commons/thumb/a/ab/Abraham_Lincoln_O-77_matte_collodion_print.jpg/1024px-Abraham_Lincoln_O-77_matte_collodion_print.jpg',
    'lincoln.jpg')
torch.hub.download_url_to_file(
    'https://user-images.githubusercontent.com/17445847/187400315-87a90ac9-d231-45d6-b377-38702bd1838f.jpg',
    'AI-generate.jpg')
torch.hub.download_url_to_file(
    'https://user-images.githubusercontent.com/17445847/187400981-8a58f7a4-ef61-42d9-af80-bc6234cef860.jpg',
    'Blake_Lively.jpg')
torch.hub.download_url_to_file(
    'https://user-images.githubusercontent.com/17445847/187401133-8a3bf269-5b4d-4432-b2f0-6d26ee1d3307.png',
    '10045.png')

# background enhancer with RealESRGAN
model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4, act_type='prelu')
netscale = 4
model_path = 'realesr-general-x4v3.pth'
half = True if torch.cuda.is_available() else False
upsampler = RealESRGANer(scale=netscale, model_path=model_path, model=model, tile=0, tile_pad=10, pre_pad=0, half=half)

# Use GFPGAN for face enhancement
face_enhancer_v3 = GFPGANer(
    model_path='GFPGANv1.3.pth', upscale=2, arch='clean', channel_multiplier=2, bg_upsampler=upsampler)
face_enhancer_v2 = GFPGANer(
    model_path='GFPGANv1.2.pth', upscale=2, arch='clean', channel_multiplier=2, bg_upsampler=upsampler)
os.makedirs('output', exist_ok=True)

def inference(img, version, scale):
    print(torch.cuda.is_available())
    img = cv2.imread(img, cv2.IMREAD_UNCHANGED)
    if len(img.shape) == 3 and img.shape[2] == 4:
        img_mode = 'RGBA'
    else:
        img_mode = None

    h, w = img.shape[0:2]
    if h < 400:
        img = cv2.resize(img, (w * 2, h * 2), interpolation=cv2.INTER_LANCZOS4)

    if version == 'v1.2':
        face_enhancer = face_enhancer_v2
    else:
        face_enhancer = face_enhancer_v3
    try:
        _, _, output = face_enhancer.enhance(img, has_aligned=False, only_center_face=False, paste_back=True)
    except RuntimeError as error:
        print('Error', error)
        print('If you encounter CUDA out of memory, try to set --tile with a smaller number.')
    else:

        extension = 'png'
    if scale != 2:
        interpolation = cv2.INTER_AREA if scale < 2 else cv2.INTER_LANCZOS4
        h, w = img.shape[0:2]
        output = cv2.resize(output, (int(w * scale /2), int(h * scale/2)), interpolation=interpolation)
    if img_mode == 'RGBA':  # RGBA images should be saved in png format
        extension = 'png'
    else:
        extension = 'jpg'
    save_path = f'output/out.{extension}'
    cv2.imwrite(save_path, output)

    output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
    return output, save_path


title = "GFPGAN: Practical Face Restoration Algorithm"
description = r"""[![GitHub Stars](https://img.shields.io/github/stars/TencentARC/GFPGAN?style=social)](https://github.com/TencentARC/GFPGAN)
Gradio demo for <a href='https://github.com/TencentARC/GFPGAN' target='_blank'><b>GFPGAN: Towards Real-World Blind Face Restoration with Generative Facial Prior</b></a>.<br>
It can be used to restore your **old photos** or improve **AI-generated faces**.<br>
To use it, simply upload your image. Please click submit only once.
"""
article = r"""<p style='text-align: center'><a href='https://arxiv.org/abs/2101.04061' target='_blank'>GFPGAN: Towards Real-World Blind Face Restoration with Generative Facial Prior</a> | <a href='https://github.com/TencentARC/GFPGAN' target='_blank'>Github Repo</a></p><center><img src='https://visitor-badge.glitch.me/badge?page_id=akhaliq_GFPGAN' alt='visitor badge'></center>

[![arXiv](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2101.04061)
[![GitHub Stars](https://img.shields.io/github/stars/TencentARC/GFPGAN?style=social)](https://github.com/TencentARC/GFPGAN)
[![download](https://img.shields.io/github/downloads/TencentARC/GFPGAN/total.svg)](https://github.com/TencentARC/GFPGAN/releases)

"""
gr.Interface(
    inference,
    [gr.inputs.Image(type="filepath", label="Input"),
     gr.inputs.Radio(['v1.2','v1.3'], type="value", default='v1.3', label='GFPGAN version'),
     gr.inputs.Number(label="Rescaling factor", default=2)],
    [gr.outputs.Image(type="numpy", label="Output (The whole image)"),
     gr.outputs.File(label="Download the output image")],
    title=title,
    description=description,
    article=article,
    examples=[['AI-generate.jpg', 'v1.3', 2], ['lincoln.jpg', 'v1.3',2], ['Blake_Lively.jpg', 'v1.3',2], ['10045.png', 'v1.3',2]]).launch()
