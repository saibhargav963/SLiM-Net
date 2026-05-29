import os
import yaml
import numpy as np
import torch

from PIL import Image

from datasets.transforms import (
get_test_transforms
)

from models.slimnet import (
SLiMNet
)

def load_config():

```
with open(
    "configs/slimnet.yaml",
    "r"
) as f:

    config = yaml.safe_load(f)

return config
```

def load_model(
checkpoint_path,
device
):

```
model = SLiMNet(
    num_classes=7
)

model.load_state_dict(

    torch.load(
        checkpoint_path,
        map_location=device
    )

)

model = model.to(device)

model.eval()

return model
```

def load_image(
image_path
):

```
transform = get_test_transforms()

image = Image.open(
    image_path
).convert("RGB")

image = transform(
    image
)

image = image.unsqueeze(0)

return image
```

def generate_change_mask(
semantic_map
):

```
background_class = 0

change_mask = (
    semantic_map != background_class
).astype(np.uint8)

return change_mask
```

def compute_change_percentage(
change_mask
):

```
changed_pixels = np.sum(
    change_mask
)

total_pixels = (
    change_mask.shape[0]
    *
    change_mask.shape[1]
)

percentage = (

    changed_pixels

    / total_pixels

) * 100.0

return percentage
```

def save_outputs(
semantic_map,
change_mask
):

```
os.makedirs(
    "results",
    exist_ok=True
)

semantic_img = Image.fromarray(
    semantic_map.astype(
        np.uint8
    )
)

semantic_img.save(
    "results/semantic_transition_map.png"
)

mask_img = Image.fromarray(
    (
        change_mask * 255
    ).astype(np.uint8)
)

mask_img.save(
    "results/change_mask.png"
)
```

def predict(
image_t1_path,
image_t2_path
):

```
device = torch.device(

    "cuda"

    if torch.cuda.is_available()

    else "cpu"

)

model = load_model(

    checkpoint_path=
    "checkpoints/slimnet_best.pth",

    device=device

)

image_t1 = load_image(
    image_t1_path
).to(device)

image_t2 = load_image(
    image_t2_path
).to(device)

with torch.no_grad():

    outputs = model(

        image_t1,

        image_t2

    )

    logits = outputs[
        "semantic_logits"
    ]

    semantic_map = torch.argmax(

        logits,

        dim=1

    )

    semantic_map = (

        semantic_map

        .squeeze()

        .cpu()

        .numpy()

    )

change_mask = generate_change_mask(
    semantic_map
)

change_percentage = (
    compute_change_percentage(
        change_mask
    )
)

save_outputs(
    semantic_map,
    change_mask
)

print(
    "\nPrediction Complete"
)

print(
    f"Change Percentage: "
    f"{change_percentage:.2f}%"
)

print(
    "Semantic transition map saved."
)

print(
    "Binary change mask saved."
)

return {

    "semantic_map":
        semantic_map,

    "change_mask":
        change_mask,

    "change_percentage":
        change_percentage

}
```

if **name** == "**main**":

```
image_t1_path = (
    "sample_data/im1.png"
)

image_t2_path = (
    "sample_data/im2.png"
)

predict(

    image_t1_path,

    image_t2_path

)
```
