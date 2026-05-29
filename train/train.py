import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torch.utils.data import random_split

from datasets.second_dataset import (
SECONDDataset
)

from datasets.transforms import (
get_train_transforms
)

from models.slimnet import (
SLiMNet
)

from utils.seed import (
set_seed
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

def create_dataloaders(
dataset_root,
batch_size
):

```
dataset = SECONDDataset(
    root_dir=dataset_root,
    split="train",
    transform=get_train_transforms()
)

train_size = int(
    0.8 * len(dataset)
)

val_size = (
    len(dataset)
    - train_size
)

train_dataset, val_dataset = random_split(

    dataset,

    [
        train_size,
        val_size
    ]

)

train_loader = DataLoader(

    train_dataset,

    batch_size=batch_size,

    shuffle=True,

    num_workers=4

)

val_loader = DataLoader(

    val_dataset,

    batch_size=batch_size,

    shuffle=False,

    num_workers=4

)

return train_loader, val_loader
```

def train_one_epoch(
model,
loader,
optimizer,
criterion,
device
):

```
model.train()

running_loss = 0.0

for batch in loader:

    image_t1 = (
        batch["image_t1"]
        .to(device)
    )

    image_t2 = (
        batch["image_t2"]
        .to(device)
    )

    label_t2 = torch.tensor(
        batch["label_t2"],
        dtype=torch.long
    ).to(device)

    optimizer.zero_grad()

    outputs = model(
        image_t1,
        image_t2
    )

    logits = outputs[
        "semantic_logits"
    ]

    loss = criterion(
        logits,
        label_t2
    )

    loss.backward()

    optimizer.step()

    running_loss += (
        loss.item()
    )

return (
    running_loss
    / len(loader)
)
```

def validate(
model,
loader,
criterion,
device
):

```
model.eval()

running_loss = 0.0

with torch.no_grad():

    for batch in loader:

        image_t1 = (
            batch["image_t1"]
            .to(device)
        )

        image_t2 = (
            batch["image_t2"]
            .to(device)
        )

        label_t2 = torch.tensor(
            batch["label_t2"],
            dtype=torch.long
        ).to(device)

        outputs = model(
            image_t1,
            image_t2
        )

        logits = outputs[
            "semantic_logits"
        ]

        loss = criterion(
            logits,
            label_t2
        )

        running_loss += (
            loss.item()
        )

return (
    running_loss
    / len(loader)
)
```

def main():

```
config = load_config()

set_seed(
    config["seed"]
)

device = torch.device(

    "cuda"

    if torch.cuda.is_available()

    else "cpu"

)

train_loader, val_loader = (

    create_dataloaders(

        dataset_root="SECOND",

        batch_size=config[
            "training"
        ]["batch_size"]

    )

)

model = SLiMNet(
    num_classes=7
).to(device)

criterion = (
    nn.CrossEntropyLoss()
)

optimizer = optim.Adam(

    model.parameters(),

    lr=config[
        "training"
    ]["learning_rate"]

)

best_val_loss = 1e10

for epoch in range(

    config["training"][
        "epochs"
    ]

):

    train_loss = train_one_epoch(

        model,

        train_loader,

        optimizer,

        criterion,

        device

    )

    val_loss = validate(

        model,

        val_loader,

        criterion,

        device

    )

    print(

        f"Epoch "
        f"{epoch+1}: "

        f"Train Loss="
        f"{train_loss:.4f} "

        f"Val Loss="
        f"{val_loss:.4f}"

    )

    if val_loss < best_val_loss:

        best_val_loss = val_loss

        os.makedirs(
            "checkpoints",
            exist_ok=True
        )

        torch.save(

            model.state_dict(),

            "checkpoints/slimnet_best.pth"

        )

        print(
            "Best model saved."
        )
```

if **name** == "**main**":

```
main()
```
