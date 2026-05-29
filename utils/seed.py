"""
seed.py

Utility functions for reproducible experiments.
Used by training, evaluation, and inference scripts.
"""

import random
import numpy as np
import torch

def set_seed(seed=42):
"""
Set random seed for reproducibility.

```
Parameters
----------
seed : int
    Random seed value.
"""

random.seed(seed)

np.random.seed(seed)

torch.manual_seed(seed)

if torch.cuda.is_available():

    torch.cuda.manual_seed(seed)

    torch.cuda.manual_seed_all(seed)

torch.backends.cudnn.deterministic = True

torch.backends.cudnn.benchmark = False

print(
    f"Random seed set to {seed}"
)
```

if **name** == "**main**":

```
set_seed(42)
```
