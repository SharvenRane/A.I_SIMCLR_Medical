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

## Evaluation Results

| Model                        | Avg Loss | Accuracy  |
|-----------------------------|----------|-----------|
| SimCLR (ResNet18)           | 0.6253   | 65.69%    |
| ResNet18 (ImageNet weights) | 0.5882   | 69.33%    |

---

## Conclusion

On this small medical dataset, the baseline ResNet18 model pretrained on ImageNet slightly outperformed the SimCLR-pretrained version. While this result doesn’t prove much, it’s likely that the amount of data used to train SimCLR wasn’t sufficient for it to learn stronger representations than ImageNet features. I still believe a SimCLR-pretrained model can outperform ImageNet weights—especially on datasets with unique characteristics that go beyond what ImageNet has already captured.
