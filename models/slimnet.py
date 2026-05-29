import torch
import torch.nn as nn

from models.shearlet import ShearletFeatureExtractor
from models.fusion import CrossShearletFusion
from models.liquid_net import LiquidNeuralNetwork
from models.error_refinement import ErrorRefinementModule

class SLiMNet(nn.Module):
"""
SLiM-Net

```
Shearlet-Liquid Misalignment Network
for Semantic Change Detection.
"""

def __init__(
    self,
    num_classes=7,
    feature_channels=128,
    hidden_dim=64
):

    super().__init__()

    # ----------------------------------
    # Shearlet Feature Extraction
    # ----------------------------------

    self.shearlet_encoder = (
        ShearletFeatureExtractor(
            in_channels=3,
            out_channels=feature_channels,
            scales=3
        )
    )

    # ----------------------------------
    # Cross-Shearlet Fusion
    # ----------------------------------

    self.fusion_module = (
        CrossShearletFusion(
            feature_channels=feature_channels
        )
    )

    # ----------------------------------
    # Liquid Neural Network
    # ----------------------------------

    self.liquid_network = (
        LiquidNeuralNetwork(
            input_channels=feature_channels,
            hidden_dim=hidden_dim
        )
    )

    # ----------------------------------
    # Error Refinement
    # ----------------------------------

    self.error_refinement = (
        ErrorRefinementModule(
            embedding_dim=hidden_dim,
            alpha=0.5,
            beta=0.5
        )
    )

    # ----------------------------------
    # Semantic Prediction Head
    # ----------------------------------

    self.classifier = nn.Sequential(

        nn.Conv2d(
            1,
            32,
            kernel_size=3,
            padding=1
        ),

        nn.BatchNorm2d(32),

        nn.ReLU(inplace=True),

        nn.Conv2d(
            32,
            num_classes,
            kernel_size=1
        )

    )

def forward(
    self,
    image_t1,
    image_t2
):

    # ==================================
    # Feature Extraction
    # ==================================

    feat_t1 = self.shearlet_encoder(
        image_t1
    )

    feat_t2 = self.shearlet_encoder(
        image_t2
    )

    # ==================================
    # Cross-Shearlet Fusion
    # ==================================

    fusion_outputs = self.fusion_module(

        feat_t1["low_frequency"],
        feat_t2["low_frequency"],

        feat_t1["high_frequency"],
        feat_t2["high_frequency"]

    )

    fused_features = (
        fusion_outputs["fused_features"]
    )

    # ==================================
    # Liquid Neural Network
    # ==================================

    temporal_embedding = (
        self.liquid_network(
            fused_features
        )
    )

    # ==================================
    # Error Refinement
    # ==================================

    error_outputs = (
        self.error_refinement(

            fusion_outputs["delta_l"],

            fusion_outputs["delta_h"],

            temporal_embedding

        )
    )

    refined_error = (
        error_outputs["refined_error"]
    )

    # ==================================
    # Semantic Prediction
    # ==================================

    semantic_logits = (
        self.classifier(
            refined_error
        )
    )

    return {

        "semantic_logits":
            semantic_logits,

        "temporal_embedding":
            temporal_embedding,

        "delta_l":
            fusion_outputs["delta_l"],

        "delta_h":
            fusion_outputs["delta_h"],

        "similarity":
            fusion_outputs["similarity"],

        "refined_error":
            refined_error

    }
```

if **name** == "**main**":

```
image_t1 = torch.randn(
    2,
    3,
    256,
    256
)

image_t2 = torch.randn(
    2,
    3,
    256,
    256
)

model = SLiMNet(
    num_classes=7
)

outputs = model(
    image_t1,
    image_t2
)

print(
    "Semantic Logits:",
    outputs["semantic_logits"].shape
)

print(
    "Temporal Embedding:",
    outputs["temporal_embedding"].shape
)

print(
    "Refined Error:",
    outputs["refined_error"].shape
)
```
