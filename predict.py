"""
This script can load our trained model to generate a fused image.
Call example: python3 predict.py ct_img_path mr_img_path
"""

import os, torchvision, torch, sys
import torch.nn as nn
from torchvision import transforms, utils
from torch.autograd import Variable
import skimage.io as io
import numpy as np
from torchvision import models
from skimage.measure import compare_ssim as ssim
from FWNet import FWNet
from utils import *
import warnings

warnings.filterwarnings("ignore")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

net = FWNet(1, 2)
net.to(device)
net.load_state_dict(torch.load('./model/fusion_model.pt'))
net.eval()

transform = transforms.Compose([
    transforms.ToTensor(),
    ]
)

img_ct = io.imread(sys.argv[1]) # '../test/case10/ct1_012.gif'
img_mr = io.imread(sys.argv[2]) # '../test/case10/mr2_012.gif'
img_ct = transform(img_ct)
img_mr = transform(img_mr)
img_ct = torch.unsqueeze(img_ct, 1)
img_mr = torch.unsqueeze(img_mr, 1)

img_fusion, oo = net(torch.cat((img_ct, img_mr), dim = 1).to(device))
oo = oo.cpu()
img_fusion = img_fusion.cpu()
img_fusion_post = post_image(img_ct[0][0], img_mr[0][0], img_fusion[0][0], chg_bg = True)

with torch.no_grad():
    io.imsave('fusion_result.tif', np.array(img_fusion_post * 255, dtype = 'uint8'))
