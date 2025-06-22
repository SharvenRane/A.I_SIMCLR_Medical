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

On this small medical dataset, the baseline ImageNet-pretrained ResNet18 slightly outperformed the SimCLR-pretrained model. This proves nothing there is a good chance that that the amount of images i used to train the simclr was probably not enough for it to learn good features to beat the ImageNet pretrained weights. I still believe simclr-pretrained model will outperform on choosing a dataset that is unique and is not already what imagenet has seen :)
