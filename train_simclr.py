# train_simclr.py

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from models.resnet_encoder import ResNetSimCLR
from utils.augmentations import SimCLRTransform
from utils.dataset_loader import ChestXrayDataset
from torch.optim import Adam
from tqdm import tqdm
import numpy as np
import random

BATCH_SIZE = 64
EPOCHS = 20
IMAGE_SIZE = 224
PROJECTION_DIM = 128
TEMPERATURE = 0.5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = "data/images_subset"
SAVE_PATH = "models/simclr_model.pth"
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

transform = SimCLRTransform(IMAGE_SIZE)
dataset = ChestXrayDataset(DATA_DIR, transform=transform)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, drop_last=True)

model = ResNetSimCLR(base_model='resnet18', out_dim=PROJECTION_DIM).to(DEVICE)

def nt_xent_loss(out_1, out_2, temperature):
    batch_size = out_1.shape[0]
    out = torch.cat([out_1, out_2], dim=0)
    sim_matrix = nn.functional.cosine_similarity(out.unsqueeze(1), out.unsqueeze(0), dim=2)
    sim_matrix = sim_matrix / temperature

    labels = torch.cat([torch.arange(batch_size) for _ in range(2)], dim=0)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float().to(DEVICE)
    mask = torch.eye(labels.shape[0], dtype=torch.bool).to(DEVICE)
    labels = labels[~mask].view(labels.shape[0], -1)
    sim_matrix = sim_matrix[~mask].view(sim_matrix.shape[0], -1)

    positives = sim_matrix[labels.bool()].view(labels.shape[0], -1)
    negatives = sim_matrix[~labels.bool()].view(sim_matrix.shape[0], -1)

    logits = torch.cat([positives, negatives], dim=1)
    labels = torch.zeros(logits.shape[0], dtype=torch.long).to(DEVICE)
    return nn.CrossEntropyLoss()(logits, labels)

optimizer = Adam(model.parameters(), lr=3e-4)

for epoch in range(EPOCHS):
    total_loss = 0
    model.train()

    for (x_i, x_j) in tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        x_i = x_i.to(DEVICE)
        x_j = x_j.to(DEVICE)

        h_i, z_i = model(x_i)
        h_j, z_j = model(x_j)

        loss = nt_xent_loss(z_i, z_j, temperature=TEMPERATURE)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(loader)
    print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")

    
    torch.save(model.state_dict(), SAVE_PATH)
