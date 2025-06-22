import torch
import torch.nn as nn
import pandas as pd
from torchvision import transforms
from torch.utils.data import DataLoader
import os

from utils.dataset_loader import CustomLabeledDataset
from models.resnet_encoder import ResNetSimCLR  

MODEL_PATH = "outputs/classifier_head_no_simclr.pth"  
USE_SIMCLR = False                          
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

df = pd.read_csv("data/Data_Entry_2017_v2020.csv")
image_dir = "data/images_subset"
df['Path'] = df['Image Index'].apply(lambda x: os.path.join(image_dir, x))
df = df[df['Path'].apply(os.path.exists)].reset_index(drop=True)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])
dataset = CustomLabeledDataset(df, image_dir, transform)
loader = DataLoader(dataset, batch_size=64, shuffle=False)

if USE_SIMCLR:
    simclr_model = ResNetSimCLR(base_model='resnet18', out_dim=128)
    simclr_model.load_state_dict(torch.load("models/simclr_model.pth", map_location=DEVICE))
    encoder = simclr_model.backbone
    encoder.fc = nn.Identity()
else:
    from torchvision import models
    encoder = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    encoder.fc = nn.Identity()

encoder.eval()
encoder.to(DEVICE)

classifier = nn.Sequential(
    nn.Linear(512, 256),
    nn.ReLU(),
    nn.Linear(256, 2)
).to(DEVICE)
classifier.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
classifier.eval()

criterion = nn.CrossEntropyLoss()
total_loss = 0
correct = 0
total = 0

with torch.no_grad():
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        features = encoder(x)
        outputs = classifier(features)
        loss = criterion(outputs, y)
        total_loss += loss.item()

        _, preds = torch.max(outputs, 1)
        correct += (preds == y).sum().item()
        total += y.size(0)

avg_loss = total_loss / len(loader)
accuracy = correct / total

print(f"\n=== Evaluation Results ===")
print(f"Average Loss: {avg_loss:.4f}")
print(f"Accuracy:     {accuracy * 100:.2f}%")
