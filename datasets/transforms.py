"""
transforms.py

Data preprocessing and augmentation
for SLiM-Net.
"""

import torchvision.transforms as T

DEFAULT_IMAGE_SIZE = 256

def get_train_transforms(
image_size=DEFAULT_IMAGE_SIZE
):

```
train_transform = T.Compose([

    T.Resize(
        (image_size, image_size),
        antialias=True
    ),

    T.RandomHorizontalFlip(
        p=0.5
    ),

    T.RandomVerticalFlip(
        p=0.5
    ),

    T.RandomRotation(
        degrees=15
    ),

    T.ColorJitter(
        brightness=0.1,
        contrast=0.1,
        saturation=0.1,
        hue=0.02
    ),

    T.ToTensor(),

    T.Normalize(
        mean=[
            0.485,
            0.456,
            0.406
        ],
        std=[
            0.229,
            0.224,
            0.225
        ]
    )

])

return train_transform
```

def get_validation_transforms(
image_size=DEFAULT_IMAGE_SIZE
):

```
val_transform = T.Compose([

    T.Resize(
        (image_size, image_size),
        antialias=True
    ),

    T.ToTensor(),

    T.Normalize(
        mean=[
            0.485,
            0.456,
            0.406
        ],
        std=[
            0.229,
            0.224,
            0.225
        ]
    )

])

return val_transform
```

def get_test_transforms(
image_size=DEFAULT_IMAGE_SIZE
):

```
test_transform = T.Compose([

    T.Resize(
        (image_size, image_size),
        antialias=True
    ),

    T.ToTensor(),

    T.Normalize(
        mean=[
            0.485,
            0.456,
            0.406
        ],
        std=[
            0.229,
            0.224,
            0.225
        ]
    )

])

return test_transform
```

if **name** == "**main**":

```
print(
    "Transforms module ready."
)
```
