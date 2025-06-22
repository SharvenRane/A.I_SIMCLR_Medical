import torch
import torch.nn as nn
import pandas as pd
from torchvision import models, transforms
from torch.utils.data import DataLoader
import os

from utils.dataset_loader import CustomLabeledDataset

BATCH_SIZE = 32
NUM_EPOCHS = 10
LR = 1e-4
NUM_CLASSES = 2
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

df = pd.read_csv('data/Data_Entry_2017_v2020.csv')
image_dir = 'data/images_subset'
df['Path'] = df['Image Index'].apply(lambda x: os.path.join(image_dir, x))
df = df[df['Path'].apply(os.path.exists)].reset_index(drop=True)

dataset = CustomLabeledDataset(df, image_dir, transform)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

encoder = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
in_features = encoder.fc.in_features
encoder.fc = nn.Identity()
encoder.eval()
encoder.to(DEVICE)

classifier = nn.Sequential(
    nn.Linear(in_features, 256),
    nn.ReLU(),
    nn.Linear(256, NUM_CLASSES)
).to(DEVICE)


criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(classifier.parameters(), lr=LR)

for epoch in range(NUM_EPOCHS):
    total_loss = 0
    classifier.train()
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        with torch.no_grad():
            features = encoder(x)
        outputs = classifier(features)
        loss = criterion(outputs, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Loss: {total_loss/len(loader):.4f}")

os.makedirs("outputs", exist_ok=True)
torch.save(classifier.state_dict(), "outputs/classifier_head_no_simclr.pth")
print("Classifier saved to outputs/classifier_head_no_simclr.pth")
