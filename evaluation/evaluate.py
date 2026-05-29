import os
import yaml
import numpy as np
import torch

from torch.utils.data import DataLoader

from datasets.second_dataset import (
SECONDDataset
)

from datasets.transforms import (
get_test_transforms
)

from models.slimnet import (
SLiMNet
)

from evaluation.metrics import (
evaluate_metrics
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

def create_test_loader():

```
dataset = SECONDDataset(

    root_dir="SECOND",

    split="test",

    transform=get_test_transforms()

)

loader = DataLoader(

    dataset,

    batch_size=1,

    shuffle=False,

    num_workers=4

)

return loader
```

def evaluate_model():

```
config = load_config()

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

test_loader = create_test_loader()

all_true = []

all_pred = []

print(
    "\nEvaluating SLiM-Net..."
)

with torch.no_grad():

    for batch in test_loader:

        image_t1 = (
            batch["image_t1"]
            .to(device)
        )

        image_t2 = (
            batch["image_t2"]
            .to(device)
        )

        label_t2 = np.array(
            batch["label_t2"]
        )

        outputs = model(

            image_t1,

            image_t2

        )

        logits = outputs[
            "semantic_logits"
        ]

        prediction = torch.argmax(

            logits,

            dim=1

        )

        prediction = (

            prediction

            .cpu()

            .numpy()

        )

        all_true.append(
            label_t2
        )

        all_pred.append(
            prediction
        )

all_true = np.concatenate(
    all_true,
    axis=0
)

all_pred = np.concatenate(
    all_pred,
    axis=0
)

results = evaluate_metrics(

    all_true,

    all_pred,

    num_classes=7

)

print("\nEvaluation Results")
print("-" * 40)

for key, value in results.items():

    print(
        f"{key}: "
        f"{value:.4f}"
    )

os.makedirs(
    "results",
    exist_ok=True
)

with open(

    "results/test_metrics.txt",

    "w"

) as f:

    for key, value in results.items():

        f.write(

            f"{key}: "
            f"{value:.6f}\n"

        )

print(
    "\nMetrics saved to "
    "results/test_metrics.txt"
)
```

if **name** == "**main**":

```
evaluate_model()
```
