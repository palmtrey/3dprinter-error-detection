import matplotlib.pyplot as plt
import torchvision.transforms as T
from dataloader import AutomationDataset
from torch.utils.data import DataLoader

train_data = AutomationDataset('/home/campalme/layer_shifting_detection/model/split_file7030.json', 
                                'train', normalize=False, augment=False)

train_loader = DataLoader(dataset=train_data,
                          batch_size = 1,
                          num_workers = 8,
                          shuffle = True)

data_iter = iter(train_loader)

fig = plt.figure()

NUM_IMAGES = 60

idx = 0

for i in data_iter:
    img = T.ToPILImage()(i[0][0])
    label = int(i[1])
    label_str = 'no_shift' if label == 0 else 'shift'

    if label == 0 and idx < NUM_IMAGES//2:
        ax = fig.add_subplot(6, 10, idx+1)
        
        ax.axes.xaxis.set_ticklabels([])
        ax.axes.yaxis.set_ticklabels([])

        # ax.set_aspect('equal')
        plt.subplots_adjust(wspace=None, hspace=None)

        ax.imshow(img) 
        idx += 1
    elif label == 1 and idx >= NUM_IMAGES//2:
        ax = fig.add_subplot(6, 10, idx+1)
        
        ax.axes.xaxis.set_ticklabels([])
        ax.axes.yaxis.set_ticklabels([])

        # ax.set_aspect('equal')
        plt.subplots_adjust(wspace=None, hspace=None)

        ax.imshow(img)
        idx += 1
    

    if idx == NUM_IMAGES:
        break
plt.axis('off')
plt.savefig('fig.png')