"""
Code for training BT/BTD models.

BT.py comparisons: [l, i, j, k, r]
BT_criteria comparisons: [c, l, i, j, k, r]
"""

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import numpy as np


# datasets 

class Comparisons(Dataset):
    """For [l, i, j, k, r] format."""
    def __init__(self, comparisons):
        self.data = comparisons

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        l, i, j, k, r = self.data[idx]
        return torch.tensor(i, dtype=torch.long), \
               torch.tensor(j, dtype=torch.long), \
               torch.tensor(k, dtype=torch.long), \
               torch.tensor(r, dtype=torch.float32)


class Comparisons_criteria(Dataset):
    """For [c, l, i, j, k, r] format."""
    def __init__(self, comparisons):
        self.data = comparisons

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        c, l, i, j, k, r = self.data[idx]
        return torch.tensor(c, dtype=torch.long), \
               torch.tensor(i, dtype=torch.long), \
               torch.tensor(j, dtype=torch.long), \
               torch.tensor(k, dtype=torch.long), \
               torch.tensor(r, dtype=torch.float32)


# ── models ──

class VectorBT(nn.Module):
    def __init__(self, num_models, d):
        super().__init__()
        self.u = nn.Embedding(num_models, d)
        self.v = nn.Embedding(num_models, d)

        nn.init.normal_(self.u.weight, mean=0.0, std=0.1)
        nn.init.normal_(self.v.weight, mean=0.0, std=0.1)

    def forward(self, i, j, k):
        u_i = self.u(i)   # shape: (batch, d)
        v_j = self.v(j)   # shape: (batch, d)
        v_k = self.v(k)   # shape: (batch, d)

        score_j = torch.sum(u_i * v_j, dim=-1)
        score_k = torch.sum(u_i * v_k, dim=-1)
        return torch.sigmoid(score_j - score_k)

class VectorBT_norm(nn.Module):
    def __init__(self, num_models, d):
        super().__init__()
        self.u = nn.Embedding(num_models, d)
        self.v = nn.Embedding(num_models, d)

        nn.init.normal_(self.u.weight, mean=0.0, std=0.1)
        nn.init.normal_(self.v.weight, mean=0.0, std=0.1)

    def forward(self, i, j, k):
        u_i = self.u(i)
        v_j = self.v(j)
        v_k = self.v(k)

        # latent strength: negative squared Euclidean distance
        score_j = -torch.sum((u_i - v_j) ** 2, dim=-1)
        score_k = -torch.sum((u_i - v_k) ** 2, dim=-1)
        return torch.sigmoid(score_j - score_k)

class VectorBT_bias(nn.Module):
    def __init__(self, num_models, d):
        super().__init__()
        self.u = nn.Embedding(num_models, d)
        self.v = nn.Embedding(num_models, d)
        self.b = nn.Embedding(num_models, 1)

        nn.init.normal_(self.u.weight, mean=0.0, std=0.1)
        nn.init.normal_(self.v.weight, mean=0.0, std=0.1)
        nn.init.normal_(self.b.weight, mean=0.0, std=0.1)

    def forward(self, i, j, k):
        u_i = self.u(i)
        v_j = self.v(j)
        v_k = self.v(k)
        b_i = self.b(i)

        score_j = torch.sum(u_i * v_j, dim=-1)
        score_k = torch.sum(u_i * v_k, dim=-1)
        return torch.sigmoid(score_j - score_k + b_i.squeeze(-1))

class VectorBTD(nn.Module):
    def __init__(self, num_models, d):
        super().__init__()
        self.u = nn.Embedding(num_models, d)
        self.v = nn.Embedding(num_models, d)
        self.log_lambda = nn.Embedding(num_models, 1)

        nn.init.normal_(self.u.weight, mean=0.0, std=0.1)
        nn.init.normal_(self.v.weight, mean=0.0, std=0.1)
        nn.init.constant_(self.log_lambda.weight, 0.0) # lambda initialized to 1

    def forward(self, i, j, k):
        u_i = self.u(i)
        v_j = self.v(j)
        v_k = self.v(k)

        score_j = torch.sum(u_i * v_j, dim=-1)
        score_k = torch.sum(u_i * v_k, dim=-1)

        log_lambda_i = self.log_lambda(i).squeeze(-1)
        tie_logit = log_lambda_i + 0.5 * (score_j + score_k)

        logits = torch.stack([tie_logit, score_j, score_k], dim=1)
        return logits

class VectorBTD_criteria(nn.Module):
    def __init__(self, num_criteria, num_models, d):
        super().__init__()
        self.num_criteria = num_criteria
        self.num_models = num_models

        self.u = nn.Embedding(num_criteria * num_models, d)
        self.v = nn.Embedding(num_models, d)
        self.log_lambda = nn.Embedding(num_criteria * num_models, 1)

        nn.init.normal_(self.u.weight, mean=0.0, std=0.1)
        nn.init.normal_(self.v.weight, mean=0.0, std=0.1)
        nn.init.constant_(self.log_lambda.weight, 0.0)

    def forward(self, c, i, j, k):
        judge = c * self.num_models + i
        u_i_c = self.u(judge)
        v_j = self.v(j)
        v_k = self.v(k)

        score_j = torch.sum(u_i_c * v_j, dim=-1)
        score_k = torch.sum(u_i_c * v_k, dim=-1)

        log_lambda_i = self.log_lambda(judge).squeeze(-1)
        tie_logit = log_lambda_i + 0.5 * (score_j + score_k)

        logits = torch.stack([tie_logit, score_j, score_k], dim=1)
        return logits

    def get_prob(self, c, i, j, k):
        judge = c * self.num_models + i
        u_i_c = self.u(judge)
        v_j = self.v(j)
        v_k = self.v(k)

        score_j = torch.sum(u_i_c * v_j, dim=-1)
        score_k = torch.sum(u_i_c * v_k, dim=-1)

        log_lambda_i = self.log_lambda(judge).squeeze(-1)
        tie_logit = log_lambda_i + 0.5 * (score_j + score_k)

        logits = [tie_logit, score_j, score_k]
        return logits


# ── training ──

def train_vector_bt(model, dataloader, lr, weight_decay, max_epochs, device, save_path=None, normalize=False, use_btd=False):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    if use_btd:
        loss_fn = nn.CrossEntropyLoss()
    else:
        loss_fn = nn.BCELoss()

    loss_history = []
    is_criteria = isinstance(model, VectorBTD_criteria)

    for epoch in range(1, max_epochs+1):
        total_loss = 0.0
        model.train()

        for batch in dataloader:
            if is_criteria:
                c, i, j, k, r = [b.to(device) for b in batch]
            else:
                i, j, k, r = [b.to(device) for b in batch]

            if use_btd:
                r = r.long()  # CrossEntropyLoss expects long tensor
                logits = model(c, i, j, k) if is_criteria else model(i, j, k)
                loss = loss_fn(logits, r)
            else:
                p = model(i, j, k)
                loss = loss_fn(p, r)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if normalize:
                with torch.no_grad():
                    model.v.weight.data = F.normalize(model.v.weight.data, p=2, dim=1)

            total_loss += loss.item() * r.size(0)

        avg_loss = total_loss / len(dataloader.dataset)
        loss_history.append(avg_loss)
    
        if len(loss_history) >= 10 and  np.average(np.abs(np.diff(loss_history[-10:]))) <= .0001:
            print('loss converged, breaking')
            break

        if epoch % 10 == 0:
            print(f"Epoch {epoch:>3d}, Loss = {avg_loss:.4f}")

    if save_path:
        model_path = save_path+"model.pt"
        torch.save(model.state_dict(), model_path)
        print(f"Model saved to {model_path}")

    return loss_history
