import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torchvision.transforms.functional as TF
from dataloader import AutomationDataset
from network import ResNeXtClassifier
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

CROP_SIZE = 800

def visualize(folder, images, results, center, out_dir):
    assert len(images) == len(results)
    
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    images = np.array(images)

    length = len(images)

    idxs = np.array([x for x in range(length) if x % 10 == 0])
    print(idxs)
    visu_images = images[idxs]
    print(visu_images)
    

    for idx in idxs:
        plt.figure()
        f, ax = plt.subplots(1) 
        ax.set_xlabel('No Shift' if results[idx] == 0 else 'Shift')
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.set_xticks([])
        ax.set_yticks([])

        img = Image.open(os.path.join(folder, images[idx]))
        img = TF.crop(img, center[1] - CROP_SIZE//2, center[0] - CROP_SIZE//2, CROP_SIZE, CROP_SIZE)
        ax.imshow(img)

        plt.savefig(os.path.join(out_dir, 'out_' + str(idx) + '.jpg'))

def eval(folder: str, model_weights: str, image_ext: str) -> None:   
    # Load the model
    model = ResNeXtClassifier(resnet_version=50).cuda(DEVICE)

    model = model.load_from_checkpoint(checkpoint_path=model_weights, resnet_version=50)
    model.eval()


    files = os.listdir(folder)
    images = [x for x in files if x.endswith(image_ext)]
    images.sort(key=lambda x: int(x.split('.')[0].split('_')[-1]))

    try:
        with open(os.path.join(folder, '_shift.json')) as f:
            split_name = json.load(f)

        with open(os.path.join(folder, '_center.json')) as f:
            center = json.load(f)

        # 0 denotes no shift, 1 denotes shift
        results = []
        labels = []

        print('\nFetching labels...')
        shift = False

        for img_path in tqdm(images):
            if shift == False:
                if img_path.replace(':', '_') != split_name:
                    labels.append(0)
                else:
                    labels.append(1)
                    shift_start = len(labels) - 1
                    shift = True
            else:
                labels.append(1)

        print('\nRunning tests...')
        predict_data = AutomationDataset(folder, 'train', predict=True)

        predict_loader = DataLoader(dataset=predict_data,
                        batch_size = 1,
                        num_workers = 8,
                        shuffle = False)

        trainer = pl.Trainer(devices=[DEVICE], accelerator='cuda')
        results = trainer.predict(model=model, dataloaders=predict_loader)

        # print(results)


        visualize(folder, images, results, center, './visualized_results/')

        difference = ['' if x == y else 'x' for x,y in zip(results, labels)]

        accuracy = difference.count('')/len(difference)

        print('\nResults for ' + str(folder) + '\n' + '-'*10)
        print('Model Accuracy: ' + str(accuracy))
    
        print('Bona Fide Labels: ' + str(labels))
        print('Predictions:  ' + str(results))
        print('Bona Fide Labels (starting from shift): ' + str(labels[shift_start:]))
        print('Predictions (starting from shift): ' + str(results[shift_start:]))


        
    except FileNotFoundError:
        print('json file missing or this is a correct folder. Skipping...')





if __name__ == '__main__':
    DEVICE = 0

    FOLDER = '/media/DATACENTER2/campalme/automation_dataset/images/ender_63'
    WEIGHTS = '/home/campalme/layer_shifting_detection/model/logs/lightning_logs/version_2/checkpoints/epoch=57-step=51562.ckpt'

    eval(FOLDER, WEIGHTS, '.jpg')