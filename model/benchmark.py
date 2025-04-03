import json
import os
from datetime import datetime

import numpy as np
import pytorch_lightning as pl
from dataloader import AutomationDataset
from network import ResNetClassifier
from torch.utils.data import DataLoader

USE_CUDA = False
DEVICE = 0
IMAGE_FOLDER = 'images/ender_91'
WEIGHTS = 'lightning_logs/version_6/checkpoints/epoch=62-step=3150.ckpt'

if __name__ == '__main__':

    if USE_CUDA:
        model = ResNetClassifier(resnet_version=18).cuda(DEVICE)
        trainer = pl.Trainer(devices=[DEVICE], accelerator='cuda')

    else:
        model = ResNetClassifier(resnet_version=18)
        trainer = pl.Trainer()

    model = model.load_from_checkpoint(checkpoint_path=WEIGHTS, resnet_version=18)
    model.eval()

    images = [x for x in os.listdir(IMAGE_FOLDER) if x.endswith('.jpg')]
    if len(images) != 0:

        durations = []

        for image in images:
            

            predict_data = AutomationDataset(os.path.join(IMAGE_FOLDER, image), 'test', predict=2)

            predict_loader = DataLoader(dataset=predict_data,
                        batch_size = 1,
                        num_workers = 1,
                        shuffle = False)

            before = datetime.now()
            results = trainer.predict(model=model, dataloaders=predict_loader)
            after = datetime.now()
            duration = after-before

            durations.append(duration.microseconds)

            with open(os.path.join(IMAGE_FOLDER, 'results.json'), 'w') as f:
                json.dump(results[0], f)

            # os.system('rm ' + str(os.path.join(IMAGE_FOLDER, image)))

            if not os.path.isdir('./images_done'):
                os.makedirs('./images_done', exist_ok=True)

        print("\n\n=== Prediction Stats ===")
        print("Mean: " + str(round(np.mean(durations)/1000, 2)) + "ms")
        print("Std dev: " + str(round(np.std(durations)/1000, 2)) + "ms")
        print("Min: " + str(round(np.min(durations)/1000, 2)) + "ms")
        print("Max: " + str(round(np.max(durations)/1000, 2)) + "ms")
        