import torch
from torch import nn, Tensor

class PositionalEncoding(nn.Module):
    def __init__(self, input_dim: int = 144, max_len: int = 5000):
        super().__init__()

        idx = 1.0 / 10000 ** (torch.arange(0, input_dim, 2) / input_dim)
        pos = torch.arange(0, max_len).reshape(max_len, 1)

        self.embedding = torch.zeros((max_len, input_dim))
        self.embedding[:, 0::2] = torch.sin(pos * idx)
        self.embedding[:, 1::2] = torch.cos(pos * idx)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x (Tensor): Tensor of shape (B, T, C)
        Returns:
            x (Tensor): Tensor of shape (B, T, C)
        """
        return x + self.embedding[None, : x.shape[1], :].to(x.device)
