# utils/loss.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class NTXentLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super(NTXentLoss, self).__init__()
        self.temperature = temperature

    def forward(self, z_i, z_j):
        N = z_i.size(0)
        z = torch.cat([z_i, z_j], dim=0)  # [2N, dim]
        z = F.normalize(z, dim=1)

        sim = torch.matmul(z, z.T) / self.temperature
        mask = torch.eye(2 * N, dtype=torch.bool).to(z.device)
        sim = sim.masked_fill(mask, -9e15)

        positives = torch.cat([torch.diag(sim, N), torch.diag(sim, -N)], dim=0)
        negatives = sim[~mask].view(2 * N, -1)

        logits = torch.cat([positives.unsqueeze(1), negatives], dim=1)
        labels = torch.zeros(2 * N).long().to(z.device)

        return F.cross_entropy(logits, labels)
