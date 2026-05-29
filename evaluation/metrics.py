import numpy as np
from sklearn.metrics import (
confusion_matrix,
cohen_kappa_score,
f1_score
)

def overall_accuracy(
y_true,
y_pred
):
"""
Overall Accuracy (OA)
"""

```
y_true = y_true.flatten()
y_pred = y_pred.flatten()

return np.mean(
    y_true == y_pred
)
```

def mean_iou(
y_true,
y_pred,
num_classes
):
"""
Mean Intersection over Union
"""

```
y_true = y_true.flatten()
y_pred = y_pred.flatten()

ious = []

for cls in range(num_classes):

    intersection = np.logical_and(
        y_true == cls,
        y_pred == cls
    ).sum()

    union = np.logical_or(
        y_true == cls,
        y_pred == cls
    ).sum()

    if union == 0:

        continue

    ious.append(
        intersection / union
    )

if len(ious) == 0:

    return 0.0

return np.mean(ious)
```

def semantic_f1(
y_true,
y_pred
):
"""
Macro F1 Score
"""

```
y_true = y_true.flatten()
y_pred = y_pred.flatten()

return f1_score(
    y_true,
    y_pred,
    average="macro"
)
```

def kappa_score(
y_true,
y_pred
):
"""
Cohen's Kappa
"""

```
y_true = y_true.flatten()
y_pred = y_pred.flatten()

return cohen_kappa_score(
    y_true,
    y_pred
)
```

def sek_score(
y_true,
y_pred,
num_classes
):
"""
SeK Metric

```
Computed as the geometric
mean of Kappa and mIoU.
"""

miou = mean_iou(
    y_true,
    y_pred,
    num_classes
)

kappa = kappa_score(
    y_true,
    y_pred
)

sek = np.sqrt(
    max(miou, 0.0)
    *
    max(kappa, 0.0)
)

return sek
```

def evaluate_metrics(
y_true,
y_pred,
num_classes
):
"""
Compute all metrics
"""

```
oa = overall_accuracy(
    y_true,
    y_pred
)

miou = mean_iou(
    y_true,
    y_pred,
    num_classes
)

f1 = semantic_f1(
    y_true,
    y_pred
)

kappa = kappa_score(
    y_true,
    y_pred
)

sek = sek_score(
    y_true,
    y_pred,
    num_classes
)

return {

    "OA": oa,

    "mIoU": miou,

    "F1": f1,

    "Kappa": kappa,

    "SeK": sek

}
```

if **name** == "**main**":

```
y_true = np.random.randint(
    0,
    7,
    (256, 256)
)

y_pred = np.random.randint(
    0,
    7,
    (256, 256)
)

results = evaluate_metrics(

    y_true,

    y_pred,

    num_classes=7

)

print(results)
```
