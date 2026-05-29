import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossShearletFusion(nn.Module):
"""
Cross-Shearlet Fusion Module

```
Computes:

    ΔL : Low-frequency difference

    ΔH : High-frequency difference

    S  : Temporal similarity

    F  : Fused representation
"""

def __init__(
    self,
    feature_channels=128
):

    super().__init__()

    self.feature_channels = feature_channels

    self.fusion_conv = nn.Sequential(

        nn.Conv2d(
            feature_channels * 3,
            feature_channels,
            kernel_size=1
        ),

        nn.BatchNorm2d(
            feature_channels
        ),

        nn.ReLU(inplace=True)

    )

def compute_low_frequency_difference(
    self,
    low_t1,
    low_t2
):

    delta_l = torch.abs(
        low_t1 - low_t2
    )

    return delta_l

def compute_high_frequency_difference(
    self,
    high_t1,
    high_t2
):

    delta_h = torch.abs(
        high_t1 - high_t2
    )

    return delta_h

def compute_temporal_similarity(
    self,
    feat_t1,
    feat_t2
):

    similarity = F.cosine_similarity(
        feat_t1,
        feat_t2,
        dim=1
    )

    similarity = similarity.unsqueeze(1)

    return similarity

def forward(
    self,
    low_t1,
    low_t2,
    high_t1,
    high_t2
):

    delta_l = self.compute_low_frequency_difference(
        low_t1,
        low_t2
    )

    delta_h = self.compute_high_frequency_difference(
        high_t1,
        high_t2
    )

    similarity = self.compute_temporal_similarity(
        high_t1,
        high_t2
    )

    similarity = similarity.repeat(
        1,
        delta_l.shape[1],
        1,
        1
    )

    fused = torch.cat(

        [
            delta_l,
            delta_h,
            similarity
        ],

        dim=1

    )

    fused = self.fusion_conv(
        fused
    )

    return {

        "delta_l": delta_l,

        "delta_h": delta_h,

        "similarity": similarity,

        "fused_features": fused

    }
```

if **name** == "**main**":

```
low1 = torch.randn(
    1,
    128,
    64,
    64
)

low2 = torch.randn(
    1,
    128,
    64,
    64
)

high1 = torch.randn(
    1,
    128,
    64,
    64
)

high2 = torch.randn(
    1,
    128,
    64,
    64
)

model = CrossShearletFusion()

outputs = model(
    low1,
    low2,
    high1,
    high2
)

print(
    "Delta L:",
    outputs["delta_l"].shape
)

print(
    "Delta H:",
    outputs["delta_h"].shape
)

print(
    "Similarity:",
    outputs["similarity"].shape
)

print(
    "Fused Features:",
    outputs["fused_features"].shape
)
```
