"""
second_dataset.py

Dataset loader for SECOND
(Semantic Change Detection Dataset)

Returns:
image_t1 : RGB image at time t1
image_t2 : RGB image at time t2
label_t1 : semantic map at time t1
label_t2 : semantic map at time t2
file_name : sample identifier
"""

import os
import numpy as np
import torch

from PIL import Image
from torch.utils.data import Dataset

class SECONDDataset(Dataset):
"""
SECOND Dataset Loader
"""

```
def __init__(
    self,
    root_dir,
    split="train",
    transform=None
):

    self.root_dir = root_dir
    self.split = split
    self.transform = transform

    self.im1_dir = os.path.join(
        root_dir,
        split,
        "im1"
    )

    self.im2_dir = os.path.join(
        root_dir,
        split,
        "im2"
    )

    self.label1_dir = os.path.join(
        root_dir,
        split,
        "label1"
    )

    self.label2_dir = os.path.join(
        root_dir,
        split,
        "label2"
    )

    for folder in [

        self.im1_dir,
        self.im2_dir,
        self.label1_dir,
        self.label2_dir

    ]:

        if not os.path.exists(folder):

            raise FileNotFoundError(
                f"Folder not found: {folder}"
            )

    self.file_names = sorted([

        f for f in os.listdir(
            self.im1_dir
        )

        if f.lower().endswith(
            ".png"
        )

    ])

    print(

        f"[SECOND] Loaded "
        f"{len(self.file_names)} samples "
        f"from '{split}' split"

    )

def __len__(self):

    return len(
        self.file_names
    )

def __getitem__(
    self,
    idx
):

    file_name = self.file_names[idx]

    image_t1 = Image.open(

        os.path.join(
            self.im1_dir,
            file_name
        )

    ).convert("RGB")

    image_t2 = Image.open(

        os.path.join(
            self.im2_dir,
            file_name
        )

    ).convert("RGB")

    label_t1 = np.array(

        Image.open(

            os.path.join(
                self.label1_dir,
                file_name
            )

        )

    ).astype(np.int64)

    label_t2 = np.array(

        Image.open(

            os.path.join(
                self.label2_dir,
                file_name
            )

        )

    ).astype(np.int64)

    label_t1 = torch.from_numpy(
        label_t1
    ).long()

    label_t2 = torch.from_numpy(
        label_t2
    ).long()

    if self.transform:

        image_t1 = self.transform(
            image_t1
        )

        image_t2 = self.transform(
            image_t2
        )

    return {

        "image_t1": image_t1,

        "image_t2": image_t2,

        "label_t1": label_t1,

        "label_t2": label_t2,

        "file_name": file_name

    }
```

def get_second_dataset(
root_dir,
split="train",
transform=None
):

```
return SECONDDataset(

    root_dir=root_dir,

    split=split,

    transform=transform

)
```

if **name** == "**main**":

```
print(
    "SECOND Dataset Loader Ready"
)
```
