import torch


class BackendTorch:
    backend = 'torch'
    torch.device('cuda')
    eig = torch.linalg.eig
