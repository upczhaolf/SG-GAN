import argparse
import time
import torchvision.utils as utils
import torch
from PIL import Image
from torch.autograd import Variable
from torchvision.transforms import ToTensor, ToPILImage
import os
from densemodel import Generator

parser = argparse.ArgumentParser(description='Test Single Image')
parser.add_argument('--upscale_factor', default=4, type=int, help='super resolution upscale factor')
parser.add_argument('--test_mode', default='GPU', type=str, choices=['GPU', 'CPU'], help='using GPU or CPU')
parser.add_argument('--image_name', default='test data/LR/RGBe1-1.png',type=str, help='test low resolution image name')
parser.add_argument('--model_name', default='srsogan.pth', type=str, help='generator model epoch name')
opt = parser.parse_args()
os.environ ["CUDA_DEVICE_ORDED"]="PCI_BUS_ID"
os.environ ["CUDA_VISIBLE_DEVICE"]='2,3'
UPSCALE_FACTOR = opt.upscale_factor
TEST_MODE = True if opt.test_mode == 'GPU' else False
IMAGE_NAME = opt.image_name
MODEL_NAME = opt.model_name

model = Generator(UPSCALE_FACTOR).eval()
if TEST_MODE:
    model.cuda()
    model.load_state_dict(torch.load('save model/' + MODEL_NAME))
else:
    model.load_state_dict(torch.load('epochs/' + MODEL_NAME, map_location=lambda storage, loc: storage))

image = Image.open(IMAGE_NAME)
image = Variable(ToTensor()(image), volatile=True).unsqueeze(0)
if TEST_MODE:
    image = image.cuda()

start = time.clock()
out = model(image)
elapsed = (time.clock() - start)
print('cost' + str(elapsed) + 's')
out_img = ToPILImage()(out[0].data.cpu())

#out_img.save('out_srf_'+str(UPSCALE_FACTOR)+'_'+IMAGE_NAME )
img_path = './output images/'
utils.save_image(out, img_path+IMAGE_NAME)