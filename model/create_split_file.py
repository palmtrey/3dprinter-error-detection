import json
import os
import random

random.seed(a=None, version=2)

DATA_DIR = '/media/DATACENTER2/campalme/automation_dataset/images/'
OUT_FN = 'split_file7030dec5.json'
TRAIN_SPLIT = 70
VAL_SPLIT = 0
TEST_SPLIT = 30
NUMS = list(range(12)) + list(range(20, 56)) + list(range(58, 91))


instances = ['ender_' + str(x) for x in NUMS]


folders = []
train_folders = []
val_folders = []
test_folders = []

for folder in os.listdir(DATA_DIR):
    if folder in instances:
        folders.append(folder)

total_folders = len(folders)

num_train_instances = int(len(folders)*TRAIN_SPLIT/100)
num_val_instances = int(len(folders)*VAL_SPLIT/100)
num_test_instances = int(len(folders)*TEST_SPLIT/100)

print('Ideal instances in train set: ' + str(num_train_instances))
print('Ideal instances in validation set: ' + str(num_val_instances))
print('Ideal instances in test set: ' + str(num_test_instances))


for i in range(num_train_instances):
    idx = random.randrange(0, len(folders))
    train_folders.append(os.path.join(DATA_DIR, folders.pop(idx)))

for i in range(num_val_instances):
    idx = random.randrange(0, len(folders))
    val_folders.append(os.path.join(DATA_DIR, folders.pop(idx)))

for i in range(num_test_instances):
    idx = random.randrange(0, len(folders))
    test_folders.append(os.path.join(DATA_DIR, folders.pop(idx)))

if len(folders) != 0:
    for instance in folders:
        train_folders.append(os.path.join(DATA_DIR, instance))
        print('Appending one extra folder to train set.')

    folders = []

out_dict = {'train': train_folders,
            'val': val_folders,
            'test': test_folders}


print('\nActual split: ')
print('\tTrain: ' + str(len(train_folders)) + ' instances (' + str(round(len(train_folders)/total_folders, 2) * 100) + '%)')
print('\tValidation: ' + str(len(val_folders)) + ' instances (' + str(round(len(val_folders)/total_folders, 2) * 100) + '%)')
print('\tTest: ' + str(len(test_folders)) + ' instances (' + str(round(len(test_folders)/total_folders, 2) * 100) + '%)')
print('Total: ' + str(total_folders) + ' instances')


with open(OUT_FN, 'w') as f:
    json.dump(out_dict, f)



