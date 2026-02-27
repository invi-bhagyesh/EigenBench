import torch
import torch.nn.functional as F

import numpy as np
from tqdm import tqdm

from .bt import VectorBT, VectorBT_bias, VectorBTD, VectorBTD_criteria

def load_vector_bt(path, num_models, d, device='cpu'):
    model = VectorBT(num_models, d).to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model

def load_vector_btd(path, num_criteria, num_models, d, device='cpu'):
    model = VectorBTD_criteria(num_criteria, num_models, d).to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model

def compute_trust_matrix(model, device='cpu'):
    U = model.u.weight.data.to(device)  # shape (N, d)
    V = model.v.weight.data.to(device)  # shape (N, d)

    # Compute raw scores S_ij = u_i · v_j
    S = U @ V.t()                       # shape (N, N)
    S = torch.exp(S)
    return S

def compute_trust_matrix_ties(model, device='cpu'):
    U = model.u.weight.data.to(device)  # shape (N, d)
    V = model.v.weight.data.to(device)  # shape (N, d)
    log_lambda = model.log_lambda.weight.data.to(device)  # shape (N, 1)
    
    # Compute raw scores s_ij = exp(u_i^T v_j)
    logits = U @ V.t()                  # shape (N, N)
    s = torch.exp(logits)               # shape (N, N)
    
    # Convert log_lambda to lambda
    lambda_i = torch.exp(log_lambda)    # shape (N, 1)
    
    # Compute tie terms
    # For S_ij = s_ij + (1/2) * lambda_i * sum_{k≠j} sqrt(s_ij * s_ik)
    # This can be rewritten as:
    # S_ij = s_ij + (1/2) * lambda_i * sqrt(s_ij) * (sum_k sqrt(s_ik) - sqrt(s_ij))
    
    sqrt_s = torch.sqrt(s)              # shape (N, N)
    sqrt_s_sum = sqrt_s.sum(dim=1, keepdim=True)  # shape (N, 1) - sum_k sqrt(s_ik) for each i
    
    # For each i,j: sqrt(s_ij) * (sum_k sqrt(s_ik) - sqrt(s_ij))
    tie_terms = sqrt_s * (sqrt_s_sum - sqrt_s)  # shape (N, N)
    
    # Apply lambda_i and the 1/2 factor
    tie_contribution = 0.5 * lambda_i * tie_terms  # shape (N, N)
    
    # Compute S_ij = s_ij + tie_contribution
    S = s + tie_contribution            # shape (N, N)
    
    # Normalize rows to get T_ij
    Z_i = S.sum(dim=1, keepdim=True)    # shape (N, 1)
    T = S / Z_i                         # shape (N, N)
    
    return T

def row_normalize(S):
    row_sums = S.sum(dim=1, keepdim=True)         # (N,1)
    C = S / row_sums
    return C

def damp_matrix(C, alpha=0):
    M = C.size(0)
    E = torch.full_like(C, 1.0 / M)
    return (1 - alpha) * C + alpha * E

def eigentrust(C, alpha=0, tol=1e-6, max_iter=1000, verbose=True):
    T = damp_matrix(C, alpha)

    # Initialize trust vector
    t = torch.full((T.size(0),), 1.0 / T.size(0), device=T.device)

    if verbose:
        for _ in tqdm(range(max_iter)):
            t_next = t @ T
            if torch.norm(t_next - t, p=1) < tol:
                break
            t = t_next
    else:
        for _ in range(max_iter):
            t_next = t @ T
            if torch.norm(t_next - t, p=1) < tol:
                break
            t = t_next

    return t_next
