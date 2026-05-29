import torch
import torch.nn as nn
import torch.nn.functional as F

class ErrorRefinementModule(nn.Module):
"""
Misalignment-Aware Error Refinement Module

```
Computes:

    Ef : Structural inconsistency

    Es : Semantic inconsistency

    E' : Refined error map
"""

def __init__(
    self,
    embedding_dim=64,
    alpha=0.5,
    beta=0.5
):

    super().__init__()

    self.alpha = alpha
    self.beta = beta

    self.semantic_projection = nn.Sequential(

        nn.Linear(
            embedding_dim,
            embedding_dim
        ),

        nn.ReLU(inplace=True),

        nn.Linear(
            embedding_dim,
            embedding_dim
        )

    )

def compute_structural_inconsistency(
    self,
    delta_l,
    delta_h
):

    ef = torch.mean(

        torch.abs(
            delta_l - delta_h
        ),

        dim=1,
        keepdim=True

    )

    return ef

def compute_semantic_inconsistency(
    self,
    temporal_embedding
):

    semantic_features = self.semantic_projection(
        temporal_embedding
    )

    es = torch.norm(

        semantic_features,

        p=2,

        dim=1,

        keepdim=True

    )

    return es

def forward(
    self,
    delta_l,
    delta_h,
    temporal_embedding
):

    ef = self.compute_structural_inconsistency(
        delta_l,
        delta_h
    )

    es = self.compute_semantic_inconsistency(
        temporal_embedding
    )

    es = es.unsqueeze(-1).unsqueeze(-1)

    es = es.expand(

        -1,
        -1,
        ef.shape[2],
        ef.shape[3]

    )

    refined_error = (

        self.alpha * ef

        +

        self.beta * es

    )

    refined_error = F.normalize(
        refined_error,
        p=2,
        dim=1
    )

    return {

        "Ef": ef,

        "Es": es,

        "refined_error": refined_error

    }
```

if **name** == "**main**":

```
delta_l = torch.randn(
    2,
    128,
    64,
    64
)

delta_h = torch.randn(
    2,
    128,
    64,
    64
)

temporal_embedding = torch.randn(
    2,
    64
)

model = ErrorRefinementModule(
    embedding_dim=64,
    alpha=0.5,
    beta=0.5
)

outputs = model(
    delta_l,
    delta_h,
    temporal_embedding
)

print(
    "Ef Shape:",
    outputs["Ef"].shape
)

print(
    "Es Shape:",
    outputs["Es"].shape
)

print(
    "Refined Error Shape:",
    outputs["refined_error"].shape
)
```
