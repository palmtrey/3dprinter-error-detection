import json
import os
import threading

import pytorch_lightning as pl
from dataloader import AutomationDataset
from network import ResNetClassifier
from torch.utils.data import DataLoader

DEVICE = 1
IMAGE_FOLDER = '/home/campalme/layer_shifting_detection/model/images'
WEIGHTS = '/home/campalme/layer_shifting_detection/model/version_8/checkpoints/epoch=65-step=59136.ckpt'

PI_SERVER = 'pi@128.153.134.177'
REMOTE_RESULTS = '/home/pi/layer_shifting_detection/model/images'

def upload_results():
    while True:
        if os.path.isfile(os.path.join(IMAGE_FOLDER, 'results.json')):
             os.system('scp ' + os.path.join(IMAGE_FOLDER, 'results.json') + ' ' + str(PI_SERVER) + ':' + str(REMOTE_RESULTS))
             os.system('rm ' + os.path.join(IMAGE_FOLDER, 'results.json'))


if __name__ == '__main__':

    model = ResNetClassifier(resnet_version=18).cuda(DEVICE)

    model = model.load_from_checkpoint(checkpoint_path=WEIGHTS, resnet_version=18)
    model.eval()

    trainer = pl.Trainer(devices=[DEVICE], accelerator='cuda')
    
    upload_thread = threading.Thread(target=upload_results)
    upload_thread.start()

    while True:
        images = [x for x in os.listdir(IMAGE_FOLDER) if x.endswith('.jpg')]

        if len(images) != 0:
            for image in images:
                predict_data = AutomationDataset(os.path.join(IMAGE_FOLDER, image), 'test', predict=2)

                predict_loader = DataLoader(dataset=predict_data,
                          batch_size = 1,
                          num_workers = 1,
                          shuffle = False)

                results = trainer.predict(model=model, dataloaders=predict_loader)
                print(results[0])

                with open(os.path.join(IMAGE_FOLDER, 'results.json'), 'w') as f:
                    json.dump(results[0], f)

                # os.system('rm ' + str(os.path.join(IMAGE_FOLDER, image)))

                if not os.path.isdir('./images_done'):
                    os.makedirs('./images_done', exist_ok=True)

                os.system('mv ' + str(os.path.join(IMAGE_FOLDER, image)) + ' images_done/')
