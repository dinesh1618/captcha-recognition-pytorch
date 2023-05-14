import torch
import numpy as np
import albumentations

from PIL import Image
from PIL import ImageFile

class ClassificationDataset:
    def __init__(self, image_paths, targets, resize=None):
        self.image_paths = image_paths
        self.targets = targets
        self.resize = resize
        # mean = (0.485, 0.456, 0.406)
        # std = (0.229, 0.224, 0.225)
        self.agu = albumentations.Compose(
            [albumentations.Normalize(always_apply=True)]
        )
        # self.agu = albumentations.Compose(
        #     [albumentations.Normalize(mean, std, max_pixel_value=255.0, always_apply=True)]
        # )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, item):
        image = Image.open(self.image_paths[item]).convert('RGB')
        target = self.targets[item]

        if self.resize is not None:
            image = image.resize(
                (self.resize[1], self.resize[0]), resample=Image.BILINEAR
            )
        image = np.array(image)
        image = self.agu(image=image)['image']
        image = np.transpose(image, (2, 0, 1)).astype(np.float32)
        return {'images': torch.tensor(image, dtype=torch.float),
                'targets': torch.tensor(target, dtype=torch.float)}

