
import pytorch_lightning as pl
from dataloader import AutomationDataset
from network import ResNeXtClassifier
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from torch.utils.data import DataLoader

DEVICE = 1
BATCH_SIZE = 10
LEARNING_RATE = 5e-3
WEIGHT_DECAY = 1e-1
EPOCHS = 100
SPLIT_FILE = '/home/campalme/layer_shifting_detection/model/split_file7030dec5.json'
OPTIMIZER = 'sgd'

tb_logger = TensorBoardLogger(save_dir="logs/")
wandb_logger = WandbLogger(project="layer_shifting_detection")



model = ResNeXtClassifier(num_classes = 2, resnet_version = 50, batch_size=BATCH_SIZE, epochs=EPOCHS,
                            optimizer = OPTIMIZER, lr = LEARNING_RATE, weight_decay=WEIGHT_DECAY, tune_fc_only=False).cuda(DEVICE)

train_data = AutomationDataset(SPLIT_FILE, 'train')

train_loader = DataLoader(dataset=train_data,
                          batch_size = BATCH_SIZE,
                          num_workers = 8,
                          shuffle = True)

test_data = AutomationDataset(SPLIT_FILE, 'test')

test_loader = DataLoader(dataset=test_data,
                         batch_size = BATCH_SIZE,
                         num_workers = 8,
                         shuffle = True)

trainer = pl.Trainer(max_epochs=EPOCHS, devices=[DEVICE], accelerator='cuda', logger=[tb_logger ,wandb_logger])
trainer.fit(model=model, train_dataloaders=train_loader,val_dataloaders=test_loader)