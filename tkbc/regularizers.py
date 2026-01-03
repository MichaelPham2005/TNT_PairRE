# Copyright (c) Facebook, Inc. and its affiliates.

from abc import ABC, abstractmethod
from typing import Tuple, Optional

import torch
from torch import nn


class Regularizer(nn.Module, ABC):
    @abstractmethod
    def forward(self, factors: Tuple[torch.Tensor]):
        pass

class N3(Regularizer):
    def __init__(self, weight: float):
        super(N3, self).__init__()
        self.weight = weight

    def forward(self, factors):
        norm = 0
        for f in factors:
            norm += self.weight * torch.sum(torch.abs(f) ** 3)
        return norm / factors[0].shape[0]


class Lambda3(Regularizer):
    def __init__(self, weight: float):
        super(Lambda3, self).__init__()
        self.weight = weight

    def forward(self, factor):
        ddiff = factor[1:] - factor[:-1]
        rank = int(ddiff.shape[1] / 2)
        diff = torch.sqrt(ddiff[:, :rank]**2 + ddiff[:, rank:]**2)**3
        return self.weight * torch.sum(diff) / (factor.shape[0] - 1)


class TemporalSmoothness(Regularizer):
    """
    Temporal smoothness regularizer for TNT-PairRE
    
    Encourages consecutive time embeddings to be similar:
    L_time = lambda * (1/(|T|-1)) * sum_{l=0}^{|T|-2} ||tau_{l+1} - tau_l||_2^2
    """
    def __init__(self, weight: float):
        super(TemporalSmoothness, self).__init__()
        self.weight = weight
    
    def forward(self, time_embeddings):
        """
        Args:
            time_embeddings: torch.Tensor of shape (n_timestamps, rank)
        
        Returns:
            loss: scalar tensor
        """
        if time_embeddings.shape[0] <= 1:
            return torch.tensor(0.0, device=time_embeddings.device)
        
        # Compute differences between consecutive timestamps
        diff = time_embeddings[1:] - time_embeddings[:-1]  # (n_timestamps-1, rank)
        
        # L2 norm squared
        squared_diff = torch.sum(diff ** 2)
        
        # Average over all consecutive pairs
        loss = self.weight * squared_diff / (time_embeddings.shape[0] - 1)
        
        return loss

