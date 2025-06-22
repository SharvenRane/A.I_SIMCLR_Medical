
from torchvision import transforms
import random

class SimCLRTransform:
    def __init__(self, input_size=224):
        color_jitter = transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(size=input_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([color_jitter], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.GaussianBlur(kernel_size=9),
            transforms.ToTensor(),
        ])

    def __call__(self, x):
        return self.transform(x), self.transform(x)  
