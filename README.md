# Chest X-Ray Classification (SimCLR vs Baseline)

This project compares two strategies for binary classification of chest X-ray images:

- **SimCLR-pretrained ResNet18 encoder**
- **Baseline ResNet18 with ImageNet weights**

Each image is labeled as:
- `0`: No Finding  
- `1`: Something Found


## Dataset

Subset of the [NIH Chest X-ray Dataset](https://www.nature.com/articles/s41597-019-0322-0):

- Metadata: `Data_Entry_2017_v2020.csv`
- Images: `images_subset/`  
- Labels derived from `Finding Labels` column:
  - "No Finding" → `0`
  - Anything else → `1`


## Folder Structure

A.I_SIMCLR_Medical/
├── evaluate.py
├── models/
│   ├── resnet_encoder.py
│   ├── projection_head.py
│   └── simclr_model.pth
├── outputs/
│   ├── classifier_head.pth
│   └── classifier_head_no_simclr.pth
├── requirements.txt
├── README.md
├── train_classifier.py              # SimCLR-based classifier
├── train_classifier_no_simclr.py    # Baseline classifier
├── train_simclr.py                  # Pretrain SimCLR encoder
└── utils/
    ├── augmentations.py
    ├── dataset_loader.py
    └── loss.py



## Evaluation Results

| Model                        | Avg Loss | Accuracy  |
|-----------------------------|----------|-----------|
| SimCLR (ResNet18)           | 0.6253   | 65.69%    |
| ResNet18 (ImageNet weights) | 0.5882   | 69.33%    |

---

## Conclusion

On this small medical dataset, the baseline ImageNet-pretrained ResNet18 slightly outperformed the SimCLR-pretrained model. This highlights how supervised transfer learning may generalize better when self-supervised pretraining lacks diversity or volume.
