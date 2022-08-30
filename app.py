import os

import cv2
import gradio as gr
import torch

from realesrgan_utils import RealESRGANer
from srvgg_arch import SRVGGNetCompact

os.system("pip freeze")
os.system(
    "wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth -P ./weights")
os.system("wget https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.2.pth -P ./weights")
os.system("wget https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth -P ./weights")

torch.hub.download_url_to_file(
    'https://upload.wikimedia.org/wikipedia/commons/thumb/a/ab/Abraham_Lincoln_O-77_matte_collodion_print.jpg/1024px-Abraham_Lincoln_O-77_matte_collodion_print.jpg',
    'lincoln.jpg')
torch.hub.download_url_to_file('https://upload.wikimedia.org/wikipedia/commons/5/50/Albert_Einstein_%28Nobel%29.png',
                               'einstein.png')
torch.hub.download_url_to_file(
    'https://upload.wikimedia.org/wikipedia/commons/thumb/9/9d/Thomas_Edison2.jpg/1024px-Thomas_Edison2.jpg',
    'edison.jpg')
torch.hub.download_url_to_file(
    'https://upload.wikimedia.org/wikipedia/commons/thumb/a/a9/Henry_Ford_1888.jpg/1024px-Henry_Ford_1888.jpg',
    'Henry.jpg')
torch.hub.download_url_to_file(
    'https://upload.wikimedia.org/wikipedia/commons/thumb/0/06/Frida_Kahlo%2C_by_Guillermo_Kahlo.jpg/800px-Frida_Kahlo%2C_by_Guillermo_Kahlo.jpg',
    'Frida.jpg')

# determine models according to model names
model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4, act_type='prelu')
netscale = 4
model_path = os.path.join('weights', 'realesr-general-x4v3.pth')

# restorer
half = True if torch.cuda.is_available() else False
upsampler = RealESRGANer(scale=netscale, model_path=model_path, model=model, tile=0, tile_pad=10, pre_pad=0, half=half)

# Use GFPGAN for face enhancement
from gfpgan_utils import GFPGANer

face_enhancer = GFPGANer(
    model_path='weights/GFPGANv1.3.pth', upscale=2, arch='clean', channel_multiplier=2, bg_upsampler=upsampler)
os.makedirs('output', exist_ok=True)


def inference(img, scale):
    img = cv2.imread(img, cv2.IMREAD_UNCHANGED)

    h, w = img.shape[0:2]
    if h < 400:
        img = cv2.resize(img, (w * 2, h * 2), interpolation=cv2.INTER_LANCZOS4)

    try:
        _, _, output = face_enhancer.enhance(img, has_aligned=False, only_center_face=False, paste_back=True)
    except RuntimeError as error:
        print('Error', error)
        print('If you encounter CUDA out of memory, try to set --tile with a smaller number.')
    else:
        extension = 'png'
    if scale != 2:
        h, w = img.shape[0:2]
        output = cv2.resize((int(w * scale /2), int(h * scale/2)), interpolation=cv2.INTER_LINEAR)
    output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
    return output


title = "GFPGAN: Practical Face Restoration Algorithm"
description = "Gradio demo for GFP-GAN: Towards Real-World Blind Face Restoration with Generative Facial Prior. To use it, simply upload your image, or click one of the examples to load them. Read more at the links below. Please click submit only once"
article = "<p style='text-align: center'><a href='https://arxiv.org/abs/2101.04061' target='_blank'>Towards Real-World Blind Face Restoration with Generative Facial Prior</a> | <a href='https://github.com/TencentARC/GFPGAN' target='_blank'>Github Repo</a></p><center><img src='https://visitor-badge.glitch.me/badge?page_id=akhaliq_GFPGAN' alt='visitor badge'></center>"
gr.Interface(
    inference, [gr.inputs.Image(type="filepath", label="Input"), gr.inputs.Number(value=2, lable="Rescaling factor")],
    gr.outputs.Image(type="numpy", label="Output (The whole image)"),
    title=title,
    description=description,
    article=article,
    examples=[['lincoln.jpg'], ['einstein.png'], ['edison.jpg'], ['Henry.jpg'], ['Frida.jpg']]).launch()
