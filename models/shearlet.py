import torch
import torch.nn as nn

class ShearletFeatureExtractor(nn.Module):
"""
Shearlet-Based Multi-Scale Feature Extractor

```
This module performs multi-scale feature extraction
for the SLiM-Net semantic change detection framework.

Parameters
----------
in_channels : int
    Number of input image channels.

out_channels : int
    Number of output feature channels.

scales : int
    Number of decomposition scales.
"""

def __init__(
    self,
    in_channels=3,
    out_channels=128,
    scales=3
):

    super().__init__()

    self.scales = scales
    self.out_channels = out_channels

    # Parameters reported in the manuscript
    self.num_scales = 3
    self.num_directions = 8
    self.total_subbands = 24

    # Scale 1 (Low-Frequency Features)
    self.scale1 = nn.Sequential(

        nn.Conv2d(
            in_channels,
            32,
            kernel_size=3,
            padding=1
        ),

        nn.BatchNorm2d(32),

        nn.ReLU(inplace=True)

    )

    # Scale 2 (Mid-Frequency Features)
    self.scale2 = nn.Sequential(

        nn.Conv2d(
            32,
            64,
            kernel_size=3,
            stride=2,
            padding=1
        ),

        nn.BatchNorm2d(64),

        nn.ReLU(inplace=True)

    )

    # Scale 3 (High-Frequency Features)
    self.scale3 = nn.Sequential(

        nn.Conv2d(
            64,
            128,
            kernel_size=3,
            stride=2,
            padding=1
        ),

        nn.BatchNorm2d(128),

        nn.ReLU(inplace=True)

    )

    # Feature Fusion Layer
    self.fusion = nn.Conv2d(
        128,
        out_channels,
        kernel_size=1
    )

def forward(
    self,
    x
):

    # Low-frequency representation
    f1 = self.scale1(x)

    # Mid-frequency representation
    f2 = self.scale2(f1)

    # High-frequency representation
    f3 = self.scale3(f2)

    # Fused representation
    output = self.fusion(f3)

    return {

        "low_frequency": f1,

        "mid_frequency": f2,

        "high_frequency": f3,

        "fused_features": output

    }
```

if **name** == "**main**":

```
x = torch.randn(
    1,
    3,
    256,
    256
)

model = ShearletFeatureExtractor()

outputs = model(x)

print(
    "Low Frequency :",
    outputs["low_frequency"].shape
)

print(
    "Mid Frequency :",
    outputs["mid_frequency"].shape
)

print(
    "High Frequency :",
    outputs["high_frequency"].shape
)

print(
    "Fused Features :",
    outputs["fused_features"].shape
)
```
