import os
from math import sqrt
import torch
import torch.nn as nn
import torch.nn.parallel
import functools
import cv2 as cv

from torchvision import transforms
from tqdm import tqdm

from options import Options
from lib.data.dataloader import load_data
from lib.models import load_model
import numpy as np
import matplotlib.pyplot as plt
from lib.models.unet_parts import *
from lib.models.unet_parts_depthwise_separable import *
from lib.models.layers import CBAM
from lib.models.swin_unet import SwinTransformerSys
from lib.models.transformer import ViT
from torch.optim import lr_scheduler
from torch.nn import init
from torchvision import utils as vutils
from lib.models.networks import NetD, weights_init, define_G, define_D, get_scheduler

image_dir = './'


def hook_func(module, input, output):
    base_name = str(module)
    index = 0
    image_name = os.path.join(image_dir, '%s_%d' % (base_name, index))
    while os.path.exists(image_name):
        index += 1
        image_name = os.path.join(image_dir, '%s_%d' % (base_name, index))

    feature_map = output.detach()
    size = feature_map.shape[1]
    size = int(sqrt(size))
    if size > 8:
        size = 8
    for i in range(size * size):
        result = feature_map[:, i:i + 1, :, :]
        vutils.save_image(result, image_name + "_channel_" + str(i) + ".png")


# if os.path.exists(image_dir) == False:
#     os.mkdir(image_dir)
opt = Options().parse()
netG = define_G(opt, norm='batch', use_dropout=False, init_type='normal')
netD = define_D(opt, norm='batch', use_sigmoid=False, init_type='normal')


path_d_0 = f"./code/weights/netD_23.pth"
path_g_0 = f"./code/weights/netG_23.pth"
path_d = f"./code/weights/bnswin.pth"
path_g = f"./code/weights/bnswing.pth"

print('>> Loading weights...')

weights_g = torch.load(path_g)['state_dict']
weights_d = torch.load(path_d)['state_dict']

print(weights_g)
exit(0)

netG.load_state_dict(weights_g)
netD.load_state_dict(weights_d)

print("done")

data = load_data(opt)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

noise = torch.empty(size=(128, 3, 32, 32), dtype=torch.float32,
                    device=device)
torch.randn((128, 3, 32, 32), out=noise)

img = cv.imread("real2.png")
print(img.shape)  # numpy数组格式为（H,W,C）
transf = transforms.ToTensor()
img_tensor = transf(img).unsqueeze(0).to(device)  # tensor数据格式是torch(C,H,W)

fake = netG(img_tensor)
res1, feat_real = netD(img_tensor)
res2, feat_fake = netD(fake)
print(res1, res2)
fake = fake.to(device)
feat_fake = feat_fake.to(device)
feat_real = feat_real.to(device)

# Calculate the anomaly score.
si = img_tensor.size()
sz = feat_real.size()
rec = (img_tensor - fake).view(si[0], si[1] * si[2] * si[3])
lat = (feat_real - feat_fake).view(sz[0], sz[1] * sz[2] * sz[3])
rec = torch.mean(torch.pow(rec, 2), dim=1)
lat = torch.mean(torch.pow(lat, 2), dim=1)
error = 0.9 * rec + 0.1 * lat
s = f'rec:{rec[0]},lat:{lat[0]},err:{error[0]}'
print(s)
vutils.save_image(fake, os.path.join(image_dir, 'real2recon.png'))
vutils.save_image(fake-img_tensor,os.path.join(image_dir,'diversity.png'))