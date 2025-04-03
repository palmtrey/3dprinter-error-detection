import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image
from torchvision import models, transforms


def predict(model: models.ResNet, input_path: str, center: tuple) -> int:
    '''Uses a loaded ResNet model and an input image and returns a prediction.
    0 - no_shift
    1 - shift
    '''

    CROP_EXTENT = 350
    OUTPUT_SIZE = (224, 224)
    FINAL_CROP = 700

    crop_boundaries = (center[0]-CROP_EXTENT, center[1]-CROP_EXTENT, center[0]+CROP_EXTENT,center[1]+CROP_EXTENT)
    # print(crop_boundaries)
    data_transforms = {
        'train': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(OUTPUT_SIZE),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    img = Image.open(input_path)

    aug = TF.crop(T.ToTensor()(img), int(center[1] - FINAL_CROP//2), int(center[0] - FINAL_CROP//2), FINAL_CROP, FINAL_CROP)
    aug = T.Resize(OUTPUT_SIZE)(aug)
    aug = T.Normalize((0.48232,), (0.23051,))(aug).unsqueeze(0)

    with torch.no_grad():
        output = model(aug)

    probabilities = torch.argmax(output, 1)

    return int(probabilities[0])