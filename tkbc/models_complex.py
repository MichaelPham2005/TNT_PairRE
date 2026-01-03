# Copyright (c) Facebook, Inc. and its affiliates.
# Complex-PairRE-TNT: Temporal Knowledge Graph Embedding with Complex Rotation

from typing import Tuple, List, Dict
import math
import torch
from torch import nn
import numpy as np

from models import TKBCModel


class ComplexPairRETNT(TKBCModel):
    """
    Complex-PairRE-TNT: Combines Complex rotation, PairRE paired relations, and TNT decomposition
    
    Key features:
    - Complex-valued embeddings for entities
    - Paired rotation vectors for head/tail (r^H, r^T)
    - TNT-style temporal/non-temporal angle decomposition
    - Score: -||e_h * r^H_{r,τ} - e_t * r^T_{r,τ}||_1 in complex space
    
    Embeddings:
    - Entities: e_x ∈ C^d (stored as 2d real values: [real, imag])
    - Relations: θ^H_{static}, θ^H_{temp}, θ^T_{static}, θ^T_{temp} ∈ R^d
    - Time: w_τ ∈ R^d
    
    Time-conditioned rotation:
    - θ^H_{r,τ} = θ^H_{static} + θ^H_{temp} · w_τ
    - r^H_{r,τ} = exp(i * θ^H_{r,τ}) = cos(θ) + i*sin(θ)
    """
    
    def __init__(
            self, sizes: Tuple[int, int, int, int], rank: int,
            no_time_emb=False, init_size: float = 1e-2
    ):
        super(ComplexPairRETNT, self).__init__()
        self.sizes = sizes
        self.rank = rank
        self.no_time_emb = no_time_emb
        
        # Entity embeddings in complex space (2 * rank: real + imaginary)
        self.entity_embeddings = nn.Embedding(sizes[0], 2 * rank, sparse=True)
        self.entity_embeddings.weight.data *= init_size
        
        # Time embeddings (real-valued modulation)
        self.time_embeddings = nn.Embedding(sizes[3], rank, sparse=True)
        self.time_embeddings.weight.data *= init_size
        
        # Relation angle embeddings (4 types for paired rotations)
        # Head-side: static + temporal
        self.rel_theta_H_static = nn.Embedding(sizes[1], rank, sparse=True)
        self.rel_theta_H_temp = nn.Embedding(sizes[1], rank, sparse=True)
        
        # Tail-side: static + temporal  
        self.rel_theta_T_static = nn.Embedding(sizes[1], rank, sparse=True)
        self.rel_theta_T_temp = nn.Embedding(sizes[1], rank, sparse=True)
        
        # Initialize angles uniformly in [0, 2π)
        nn.init.uniform_(self.rel_theta_H_static.weight, 0, 2 * np.pi)
        nn.init.uniform_(self.rel_theta_H_temp.weight, -np.pi, np.pi)
        nn.init.uniform_(self.rel_theta_T_static.weight, 0, 2 * np.pi)
        nn.init.uniform_(self.rel_theta_T_temp.weight, -np.pi, np.pi)
    
    @staticmethod
    def has_time():
        return True
    
    def _angle_to_complex(self, theta: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convert angles to complex rotation: exp(i*θ) = cos(θ) + i*sin(θ)
        
        Args:
            theta: (batch, rank) angles
        
        Returns:
            (cos_theta, sin_theta): tuple of (batch, rank) tensors
        """
        return torch.cos(theta), torch.sin(theta)
    
    def _complex_multiply(
        self, 
        a_real: torch.Tensor, a_imag: torch.Tensor,
        b_real: torch.Tensor, b_imag: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Complex multiplication: (a_r + i*a_i) * (b_r + i*b_i)
        = (a_r*b_r - a_i*b_i) + i*(a_r*b_i + a_i*b_r)
        """
        real = a_real * b_real - a_imag * b_imag
        imag = a_real * b_imag + a_imag * b_real
        return real, imag
    
    def score(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute scores for given quadruples (h, r, t, τ)
        
        Bilinear form: Re(<e_h * r^H, e_t * conj(r^T)>)
        = sum(lhs_real * rhs_real + lhs_imag * rhs_imag)
        
        Args:
            x: (batch, 4) containing (h, r, t, τ)
        
        Returns:
            scores: (batch, 1)
        """
        # Extract embeddings
        e_h = self.entity_embeddings(x[:, 0])  # (batch, 2*rank)
        e_t = self.entity_embeddings(x[:, 2])  # (batch, 2*rank)
        w_tau = self.time_embeddings(x[:, 3])  # (batch, rank)
        
        # Split complex embeddings into real/imaginary
        e_h_real, e_h_imag = e_h[:, :self.rank], e_h[:, self.rank:]
        e_t_real, e_t_imag = e_t[:, :self.rank], e_t[:, self.rank:]
        
        # Get relation angles
        theta_H_s = self.rel_theta_H_static(x[:, 1])  # (batch, rank)
        theta_H_t = self.rel_theta_H_temp(x[:, 1])    # (batch, rank)
        theta_T_s = self.rel_theta_T_static(x[:, 1])  # (batch, rank)
        theta_T_t = self.rel_theta_T_temp(x[:, 1])    # (batch, rank)
        
        # Time-conditioned angles: θ_{r,τ} = θ_static + θ_temp * w_τ
        theta_H = theta_H_s + theta_H_t * w_tau
        theta_T = theta_T_s + theta_T_t * w_tau
        
        # Convert to complex rotations
        r_H_real, r_H_imag = self._angle_to_complex(theta_H)
        r_T_real, r_T_imag = self._angle_to_complex(theta_T)
        
        # Apply rotations: lhs = e_h * r^H
        lhs_real, lhs_imag = self._complex_multiply(
            e_h_real, e_h_imag, r_H_real, r_H_imag
        )
        
        # Apply rotations: rhs = e_t * conj(r^T) = e_t * (cos(-θ_T) + i*sin(-θ_T))
        # conj(r^T) means we use (r_T_real, -r_T_imag)
        rhs_real, rhs_imag = self._complex_multiply(
            e_t_real, e_t_imag, r_T_real, -r_T_imag  # Note: conjugate
        )
        
        # Bilinear form: Re(<lhs, rhs>) = lhs_real*rhs_real + lhs_imag*rhs_imag
        score = torch.sum(
            lhs_real * rhs_real + lhs_imag * rhs_imag,
            dim=1, keepdim=True
        )
        
        return score
    
    def forward(self, x: torch.Tensor):
        """
        Forward pass for training (1-vs-all)
        
        Bilinear form allows efficient matrix multiplication:
        scores = lhs @ rhs.T where lhs = e_h * r^H, rhs = e_t * conj(r^T)
        
        Args:
            x: (batch, 4) containing (h, r, t, τ)
        
        Returns:
            scores: (batch, n_entities)
            factors: tuple for regularization
            time: time embeddings for temporal regularization
        """
        batch_size = x.shape[0]
        
        # Extract embeddings
        e_h = self.entity_embeddings(x[:, 0])  # (batch, 2*rank)
        e_t = self.entity_embeddings(x[:, 2])  # (batch, 2*rank)
        w_tau = self.time_embeddings(x[:, 3])  # (batch, rank)
        
        # Split complex embeddings
        e_h_real, e_h_imag = e_h[:, :self.rank], e_h[:, self.rank:]
        e_t_real, e_t_imag = e_t[:, :self.rank], e_t[:, self.rank:]
        
        # Get relation angles
        theta_H_s = self.rel_theta_H_static(x[:, 1])
        theta_H_t = self.rel_theta_H_temp(x[:, 1])
        theta_T_s = self.rel_theta_T_static(x[:, 1])
        theta_T_t = self.rel_theta_T_temp(x[:, 1])
        
        # Time-conditioned angles
        theta_H = theta_H_s + theta_H_t * w_tau
        theta_T = theta_T_s + theta_T_t * w_tau
        
        # Convert to complex rotations
        r_H_real, r_H_imag = self._angle_to_complex(theta_H)
        r_T_real, r_T_imag = self._angle_to_complex(theta_T)
        
        # Apply head-side rotation: lhs = e_h * r^H
        lhs_real, lhs_imag = self._complex_multiply(
            e_h_real, e_h_imag, r_H_real, r_H_imag
        )
        
        # Get all entity embeddings
        all_entities = self.entity_embeddings.weight  # (n_entities, 2*rank)
        all_e_real = all_entities[:, :self.rank].t()  # (rank, n_entities) - transposed!
        all_e_imag = all_entities[:, self.rank:].t()  # (rank, n_entities)
        
        # Apply tail-side rotation with conjugate: rhs = all_e * conj(r^T)
        # For conjugate: multiply by (r_T_real, -r_T_imag)
        # But we need to broadcast r_T over all entities
        # Instead, we compute: lhs @ (conj(r^T) * all_e).T
        
        # Expand r_T for broadcasting
        r_T_real_exp = r_T_real.unsqueeze(2)  # (batch, rank, 1)
        r_T_imag_exp = r_T_imag.unsqueeze(2)  # (batch, rank, 1)
        
        # all_e: (rank, n_entities) -> expand to (1, rank, n_entities)
        all_e_real_exp = all_e_real.unsqueeze(0)
        all_e_imag_exp = all_e_imag.unsqueeze(0)
        
        # Compute: all_e * conj(r^T) = all_e * (r_T_real - i*r_T_imag)
        # Real part: all_e_real * r_T_real + all_e_imag * r_T_imag
        # Imag part: all_e_imag * r_T_real - all_e_real * r_T_imag
        rhs_real = all_e_real_exp * r_T_real_exp + all_e_imag_exp * r_T_imag_exp  # (batch, rank, n_entities)
        rhs_imag = all_e_imag_exp * r_T_real_exp - all_e_real_exp * r_T_imag_exp  # (batch, rank, n_entities)
        
        # Bilinear form: <lhs, rhs> = lhs_real * rhs_real + lhs_imag * rhs_imag
        # lhs: (batch, rank) -> (batch, rank, 1) for broadcasting
        lhs_real_exp = lhs_real.unsqueeze(2)
        lhs_imag_exp = lhs_imag.unsqueeze(2)
        
        # scores: sum over rank dimension
        scores = torch.sum(
            lhs_real_exp * rhs_real + lhs_imag_exp * rhs_imag,
            dim=1  # sum over rank
        )  # (batch, n_entities)
        
        # Factors for regularization
        entity_magnitude_h = torch.sqrt(e_h_real ** 2 + e_h_imag ** 2 + 1e-10)
        entity_magnitude_t = torch.sqrt(e_t_real ** 2 + e_t_imag ** 2 + 1e-10)
        
        # Regularize temporal modulation (w_tau effects)
        temporal_effect_H = torch.abs(theta_H_t * w_tau)
        temporal_effect_T = torch.abs(theta_T_t * w_tau)
        
        factors = (
            entity_magnitude_h,
            temporal_effect_H + temporal_effect_T,
            w_tau,
            entity_magnitude_t
        )
        
        # Time embeddings for temporal regularization
        time = self.time_embeddings.weight[:-1] if self.no_time_emb else self.time_embeddings.weight
        
        return scores, factors, time
    
    def forward_over_time(self, x: torch.Tensor):
        """
        Compute scores over all timestamps for given (h, r, t)
        
        Args:
            x: (batch, 4+) containing at least (h, r, t)
        
        Returns:
            scores: (batch, n_timestamps)
        """
        # Extract embeddings
        e_h = self.entity_embeddings(x[:, 0])
        e_t = self.entity_embeddings(x[:, 2])
        
        e_h_real, e_h_imag = e_h[:, :self.rank], e_h[:, self.rank:]
        e_t_real, e_t_imag = e_t[:, :self.rank], e_t[:, self.rank:]
        
        # Get relation angles
        theta_H_s = self.rel_theta_H_static(x[:, 1])
        theta_H_t = self.rel_theta_H_temp(x[:, 1])
        theta_T_s = self.rel_theta_T_static(x[:, 1])
        theta_T_t = self.rel_theta_T_temp(x[:, 1])
        
        # Get all time embeddings
        all_w_tau = self.time_embeddings.weight  # (n_timestamps, rank)
        
        # Expand for broadcasting
        theta_H_s_exp = theta_H_s.unsqueeze(1)  # (batch, 1, rank)
        theta_H_t_exp = theta_H_t.unsqueeze(1)
        theta_T_s_exp = theta_T_s.unsqueeze(1)
        theta_T_t_exp = theta_T_t.unsqueeze(1)
        w_tau_exp = all_w_tau.unsqueeze(0)      # (1, n_timestamps, rank)
        
        # Time-conditioned angles for all timestamps
        theta_H_all = theta_H_s_exp + theta_H_t_exp * w_tau_exp  # (batch, n_timestamps, rank)
        theta_T_all = theta_T_s_exp + theta_T_t_exp * w_tau_exp
        
        # Convert to complex rotations
        r_H_real, r_H_imag = self._angle_to_complex(theta_H_all)
        r_T_real, r_T_imag = self._angle_to_complex(theta_T_all)
        
        # Expand entity embeddings
        e_h_real_exp = e_h_real.unsqueeze(1)  # (batch, 1, rank)
        e_h_imag_exp = e_h_imag.unsqueeze(1)
        e_t_real_exp = e_t_real.unsqueeze(1)
        e_t_imag_exp = e_t_imag.unsqueeze(1)
        
        # Apply rotations
        lhs_real, lhs_imag = self._complex_multiply(
            e_h_real_exp, e_h_imag_exp, r_H_real, r_H_imag
        )
        rhs_real, rhs_imag = self._complex_multiply(
            e_t_real_exp, e_t_imag_exp, r_T_real, r_T_imag
        )
        
        # Compute scores
        diff_real = lhs_real - rhs_real
        diff_imag = lhs_imag - rhs_imag
        
        scores = -(torch.abs(diff_real).sum(dim=2) + 
                  torch.abs(diff_imag).sum(dim=2))
        
        return scores
    
    def get_rhs(self, chunk_begin: int, chunk_size: int):
        """Get entity embeddings for ranking"""
        return self.entity_embeddings.weight[
            chunk_begin:chunk_begin + chunk_size
        ].transpose(0, 1)
    
    def get_queries(self, queries: torch.Tensor):
        """
        For ComplexPairRETNT with L1 distance, we override get_ranking
        This method is not used for actual scoring
        """
        # Extract embeddings
        e_h = self.entity_embeddings(queries[:, 0])
        w_tau = self.time_embeddings(queries[:, 3])
        
        e_h_real, e_h_imag = e_h[:, :self.rank], e_h[:, self.rank:]
        
        theta_H_s = self.rel_theta_H_static(queries[:, 1])
        theta_H_t = self.rel_theta_H_temp(queries[:, 1])
        
        theta_H = theta_H_s + theta_H_t * w_tau
        r_H_real, r_H_imag = self._angle_to_complex(theta_H)
        
        lhs_real, lhs_imag = self._complex_multiply(
            e_h_real, e_h_imag, r_H_real, r_H_imag
        )
        
        return torch.cat([lhs_real, lhs_imag], dim=1)
    
    def get_ranking(
            self, queries: torch.Tensor,
            filters: Dict[Tuple[int, int, int], List[int]],
            batch_size: int = 1000, chunk_size: int = -1
    ):
        """
        Custom get_ranking for ComplexPairRETNT with bilinear form
        Now much faster with matrix multiplication!
        """
        if chunk_size < 0:
            chunk_size = self.sizes[2]
        
        ranks = torch.ones(len(queries))
        
        with torch.no_grad():
            b_begin = 0
            while b_begin < len(queries):
                these_queries = queries[b_begin:b_begin + batch_size]
                
                # Extract embeddings
                e_h = self.entity_embeddings(these_queries[:, 0])
                w_tau = self.time_embeddings(these_queries[:, 3])
                
                e_h_real, e_h_imag = e_h[:, :self.rank], e_h[:, self.rank:]
                
                # Get relation angles
                theta_H_s = self.rel_theta_H_static(these_queries[:, 1])
                theta_H_t = self.rel_theta_H_temp(these_queries[:, 1])
                theta_T_s = self.rel_theta_T_static(these_queries[:, 1])
                theta_T_t = self.rel_theta_T_temp(these_queries[:, 1])
                
                # Time-conditioned angles
                theta_H = theta_H_s + theta_H_t * w_tau
                theta_T = theta_T_s + theta_T_t * w_tau
                
                r_H_real, r_H_imag = self._angle_to_complex(theta_H)
                r_T_real, r_T_imag = self._angle_to_complex(theta_T)
                
                # Compute lhs = e_h * r^H
                lhs_real, lhs_imag = self._complex_multiply(
                    e_h_real, e_h_imag, r_H_real, r_H_imag
                )
                
                # Get target scores
                targets = self.score(these_queries).squeeze(1)
                
                # Get all entities
                all_entities = self.entity_embeddings.weight
                all_e_real = all_entities[:, :self.rank].t()  # (rank, n_entities)
                all_e_imag = all_entities[:, self.rank:].t()
                
                # Expand for broadcasting
                r_T_real_exp = r_T_real.unsqueeze(2)  # (batch, rank, 1)
                r_T_imag_exp = r_T_imag.unsqueeze(2)
                all_e_real_exp = all_e_real.unsqueeze(0)  # (1, rank, n_entities)
                all_e_imag_exp = all_e_imag.unsqueeze(0)
                
                # Compute rhs = all_e * conj(r^T)
                rhs_real = all_e_real_exp * r_T_real_exp + all_e_imag_exp * r_T_imag_exp
                rhs_imag = all_e_imag_exp * r_T_real_exp - all_e_real_exp * r_T_imag_exp
                
                # Bilinear scores
                lhs_real_exp = lhs_real.unsqueeze(2)
                lhs_imag_exp = lhs_imag.unsqueeze(2)
                
                scores = torch.sum(
                    lhs_real_exp * rhs_real + lhs_imag_exp * rhs_imag,
                    dim=1
                )  # (batch, n_entities)
                
                # Count entities scoring better than target
                targets_expanded = targets.unsqueeze(1)
                ranks[b_begin:b_begin + len(these_queries)] += torch.sum(
                    (scores > targets_expanded).float(), dim=1
                ).cpu()
                
                b_begin += batch_size
        
        return ranks
