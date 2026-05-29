# SLiM-Net: Shearlet-Liquid Misalignment Network for Semantic Change Detection

## Overview

SLiM-Net is a semantic change detection framework that integrates:

* Shearlet-based multi-scale feature extraction
* Cross-temporal feature fusion
* Liquid Neural Network (LNN) temporal modeling
* Misalignment-aware error refinement
* Semantic transition prediction

The framework is designed for robust semantic change detection in multi-temporal remote sensing imagery.

---

## Architecture

The overall pipeline is:

Input Images (T1, T2)

↓

Shearlet-Based Multi-Scale Feature Extraction

↓

Cross-Shearlet Fusion

↓

Liquid Neural Network

↓

Error Refinement

↓

Semantic Change Prediction

↓

Semantic Transition Map + Change Mask + Change Percentage

---

## Repository Structure

```text
SLiM-Net/
│
├── configs/
├── datasets/
├── models/
├── train/
├── evaluation/
├── inference/
├── utils/
│
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

---

## Datasets

### SECOND Dataset

Semantic Change Detection Dataset (SECOND)

Dataset Link:

https://captain-whu.github.io/SCD/

### SEN12MS Dataset

SEN12MS Remote Sensing Dataset

Dataset Link:

https://mediatum.ub.tum.de/1474000

Datasets are publicly available and must be downloaded separately.

---

## Installation

Create a Python environment and install dependencies:

```bash
pip install -r requirements.txt
```

---

## Configuration

Model parameters are defined in:

```text
configs/slimnet.yaml
```

Key parameters include:

* Image Size: 256 × 256
* Number of Scales: 3
* Number of Directions: 8
* Total Subbands: 24
* Hidden Dimension: 64
* Batch Size: 8
* Learning Rate: 0.0001
* Epochs: 100

---

## Training

Run:

```bash
python train/train.py
```

The best model checkpoint will be stored in:

```text
checkpoints/slimnet_best.pth
```

---

## Evaluation

Run:

```bash
python evaluation/evaluate.py
```

Reported metrics:

* Overall Accuracy (OA)
* Mean IoU (mIoU)
* F1 Score
* Kappa
* SeK

---

## Inference

Run:

```bash
python inference/predict.py
```

Outputs:

* Semantic Transition Map
* Binary Change Mask
* Change Percentage

Results are stored in:

```text
results/
```

---

## Reproducibility

A fixed random seed is used throughout all experiments.

Configuration files, dataset loaders, training scripts, evaluation scripts, and inference modules are provided to facilitate reproducible experimentation.

---

## Code Availability

The source code and reproducibility resources for SLiM-Net are publicly available through Zenodo:

https://doi.org/10.5281/zenodo.20441944

## Citation

If you use this repository in your research, please cite:

```bibtex
@article{SLiMNet2026,
  title={SLiM-Net: Shearlet-Liquid Misalignment Network for Semantic Change Detection},
  author={Author(s)},
  journal={Array},
  year={2026}
}
```

---

## License

This project is released under the MIT License.
