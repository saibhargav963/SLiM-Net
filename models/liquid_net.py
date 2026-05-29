import torch
import torch.nn as nn

class LiquidNeuron(nn.Module):
"""
Liquid Neuron

```
Continuous adaptive state update
used for temporal feature modeling.
"""

def __init__(
    self,
    input_dim,
    hidden_dim=64,
    alpha=0.5
):

    super().__init__()

    self.hidden_dim = hidden_dim
    self.alpha = alpha

    self.input_projection = nn.Linear(
        input_dim,
        hidden_dim
    )

    self.state_projection = nn.Linear(
        hidden_dim,
        hidden_dim
    )

    self.activation = nn.Tanh()

def forward(
    self,
    x,
    h_prev
):

    input_term = self.input_projection(
        x
    )

    state_term = self.state_projection(
        h_prev
    )

    h_new = (

        (1.0 - self.alpha)
        * h_prev

        +

        self.alpha
        * self.activation(
            input_term + state_term
        )

    )

    return h_new
```

class LiquidNeuralNetwork(nn.Module):
"""
Liquid Neural Network

```
Generates temporal embedding Z
from fused feature representations.
"""

def __init__(
    self,
    input_channels=128,
    hidden_dim=64
):

    super().__init__()

    self.hidden_dim = hidden_dim

    self.global_pool = nn.AdaptiveAvgPool2d(
        (1, 1)
    )

    self.liquid_neuron = LiquidNeuron(
        input_dim=input_channels,
        hidden_dim=hidden_dim,
        alpha=0.5
    )

def forward(
    self,
    fused_features
):

    batch_size = fused_features.size(0)

    pooled = self.global_pool(
        fused_features
    )

    pooled = pooled.view(
        batch_size,
        -1
    )

    h0 = torch.zeros(
        batch_size,
        self.hidden_dim,
        device=fused_features.device
    )

    temporal_embedding = self.liquid_neuron(
        pooled,
        h0
    )

    return temporal_embedding
```

if **name** == "**main**":

```
x = torch.randn(
    2,
    128,
    64,
    64
)

model = LiquidNeuralNetwork(
    input_channels=128,
    hidden_dim=64
)

z = model(x)

print(
    "Temporal Embedding Shape:",
    z.shape
)
```
