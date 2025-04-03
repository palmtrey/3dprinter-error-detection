import json
import os
import time

import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image

torch.manual_seed(time.time())

IMG_FN = '/media/DATACENTER2/campalme/automation_dataset/images/ender_1/ender_1_150.jpg'
CROP_SIZE = 800
FINAL_CROP = 700
OUTPUT_SIZE = (224, 224)

img = Image.open(IMG_FN)

with open(os.path.join(os.path.dirname(IMG_FN), '_center.json'), 'r') as f:
    center = json.load(f)

print(center)

img.save('test.jpg')

aug = TF.crop(T.ToTensor()(img), int(center[1] - CROP_SIZE//2), int(center[0] - CROP_SIZE//2), CROP_SIZE, CROP_SIZE)

aug = T.Resize(OUTPUT_SIZE)(aug)

aug = T.ToPILImage()(aug)

aug.save('test_cropped.jpg')

img_fn = IMG_FN
img_folder = os.path.dirname(img_fn)


if os.path.isfile(os.path.join(img_folder, '_shift.json')):
    with open(os.path.join(img_folder, '_shift.json'), 'r') as f:
        shift_name = json.load(f)

    img_num = int(img_fn.split('/')[-1].split('.')[0].split('_')[-1])
    shift_num = int(shift_name.split('.')[0].split('_')[-1])
    
    print('img_num: ' + str(img_num))
    print('shift_num: ' + str(shift_num))

    print(shift_name)

    if img_num >= shift_num:
        label = 1
    else:
        label = 0
else:
    label = 0

print(label)