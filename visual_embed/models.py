import sys
import os
from PIL import Image
import requests
import numpy as np
import torch
import torch.nn as nn
from timm.models.vision_transformer import PatchEmbed, Block
from torchvision.transforms import functional as TF
from functools import partial
from tqdm import tqdm
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CocoDetection
import torch.optim as optim
from . import models_mae


# Load an image
img_url = 'https://user-images.githubusercontent.com/11435359/147738734-196fd92f-9260-48d5-ba7e-bf103d29364d.jpg'  # fox, from ILSVRC2012_val_00046145
# img_url = 'https://user-images.githubusercontent.com/11435359/147743081-0428eecf-89e5-4e07-8da5-a30fd73cc0ba.jpg'  # cucumber, from ILSVRC2012_val_00047851
img = Image.open(requests.get(img_url, stream=True).raw)
img = img.resize((224, 224))
img = np.array(img) / 255.

assert img.shape == (224, 224, 3)

# ImageNet mean and std
imagenet_mean = np.array([0.485, 0.456, 0.406])
imagenet_std = np.array([0.229, 0.224, 0.225])

# Normalize by ImageNet mean and std
img = (img - imagenet_mean) / imagenet_std

# Ensure image data is in the range [0, 1] for visualization
img_for_display = img * imagenet_std + imagenet_mean
img_for_display = np.clip(img_for_display, 0, 1)

class MAEEncoder(nn.Module):
    def __init__(self, patch_embed, cls_token, pos_embed, blocks, norm):
        super(MAEEncoder, self).__init__()
        self.patch_embed = patch_embed
        self.cls_token = cls_token
        self.pos_embed = pos_embed
        self.blocks = blocks
        self.norm = norm

    def forward(self, x):
        x = self.patch_embed(x)
        x = x + self.pos_embed[:, 1:, :]
        cls_tokens = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_tokens.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x

def prepare_model(chkpt_dir='mae_visualize_vit_large.pth', arch='mae_vit_large_patch16', only_encoder=True):
    model = getattr(models_mae, arch)()
    checkpoint = torch.load(chkpt_dir, map_location='cpu')
    model.load_state_dict(checkpoint['model'], strict=False)
    
    if only_encoder:
        encoder = MAEEncoder(
            model.patch_embed,
            model.cls_token,
            model.pos_embed,
            model.blocks,
            model.norm
        )
        # Freeze parameters
        for param in encoder.parameters():
            param.requires_grad = False
            
        return encoder
    
    return model

def download_mae_gan_params():
    url = "https://dl.fbaipublicfiles.com/mae/visualize/mae_visualize_vit_large.pth"
    response = requests.get(url)
    with open("mae_visualize_vit_large.pth", "wb") as file:
        file.write(response.content)

if __name__ == "__main__":
    # Download model parameters
    download_mae_gan_params()
    
    # Prepare the encoder
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_mae_gan_encoder = prepare_model('mae_visualize_vit_large.pth', 'mae_vit_large_patch16', only_encoder=True).to(device)
    
    # Convert image to tensor
    img_tensor = torch.tensor(img).permute(2, 0, 1).unsqueeze(0).to(device, dtype=torch.float32)
    
    # Pass through encoder
    encoded_img = model_mae_gan_encoder(img_tensor)
    
    print(encoded_img.shape)  # Check the output shape
