import json
import os

import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset

CROP_SIZE = 800
FINAL_CROP = 700
OUTPUT_SIZE = (224, 224)
IMG_EXT = '.jpg'


class AutomationDataset(Dataset):
    def __init__(
        self,
        data_dir,
        phase,
        predict=0,
        normalize=True,
        augment=True,
    ):
        self.data_dir = data_dir
        self.phase = phase
        self.normalize = normalize
        self.augment = augment

        if predict == 0:
            # For regular training. Loads data from json file.
            self.folder_list = self._read_data(data_dir)[phase]
            self.data_list = []
            for folder in self.folder_list:
                self.data_list += [os.path.join(folder, x) for x in os.listdir(folder) if x.endswith(IMG_EXT)]
            self.data_list.sort(key=lambda x: int(x.split('.')[0].split('_')[-1]))
        elif predict == 1:
            # For batch predicting. Loads data from given folder.
            self.folder_list = [data_dir]
            self.data_list = []
            for folder in self.folder_list:
                self.data_list += [os.path.join(folder, x) for x in os.listdir(folder) if x.endswith(IMG_EXT)]
            self.data_list.sort(key=lambda x: int(x.split('.')[0].split('_')[-1]))
        elif predict == 2:
            # For single image predicting. Loads one image given in self.data_dir
            self.data_list = [self.data_dir]
        
        

        # print(self.data_list)

    def _read_data(self, data_fn):
        with open(self.data_dir, "r") as f:
            split_file = json.load(f)
        return split_file

    def __getitem__(self, index):
        
        img_fn = self.data_list[index]
        img_folder = os.path.dirname(img_fn)
        img = Image.open(img_fn)

        # Get instance center
        with open(os.path.join(img_folder, '_center.json'), 'r') as f:
            center = json.load(f)

        
        # print(center)
        if self.phase == 'train' and self.augment:
            aug = T.RandomRotation(degrees=(-10, 10))(T.ToTensor()(img))
            aug = TF.crop(T.ToPILImage()(aug), center[1] - CROP_SIZE//2, center[0] - CROP_SIZE//2, CROP_SIZE, CROP_SIZE)
            aug = T.RandomCrop(FINAL_CROP)(T.ToTensor()(aug))
            aug = T.Resize(OUTPUT_SIZE)(aug)
            if self.normalize:
                aug = T.Normalize((0.48232,), (0.23051,))(aug)
        else:
            aug = TF.crop(T.ToTensor()(img), int(center[1] - FINAL_CROP//2), int(center[0] - FINAL_CROP//2), FINAL_CROP, FINAL_CROP)
            aug = T.Resize(OUTPUT_SIZE)(aug)
            if self.normalize:
                aug = T.Normalize((0.48232,), (0.23051,))(aug)
        
        # Labeling data
        if os.path.isfile(os.path.join(img_folder, '_shift.json')):
            with open(os.path.join(img_folder, '_shift.json'), 'r') as f:
                shift_fn = json.load(f)

            img_num = int(img_fn.split('/')[-1].split('.')[0].split('_')[-1])
            shift_num = int(shift_fn.split('.')[0].split('_')[-1])

            if img_num >= shift_num:
                label = 1
            else:
                label = 0
        else:
            label = 0
        return (aug, label)

    def __len__(self):
        return len(self.data_list)



        