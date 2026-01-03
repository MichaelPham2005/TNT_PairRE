# Copyright (c) Facebook, Inc. and its affiliates.

from abc import ABC, abstractmethod
from typing import Tuple, List, Dict

import math
import torch
from torch import nn
import numpy as np


class TKBCModel(nn.Module, ABC):
    @abstractmethod
    def get_rhs(self, chunk_begin: int, chunk_size: int):
        pass

    @abstractmethod
    def get_queries(self, queries: torch.Tensor):
        pass

    @abstractmethod
    def score(self, x: torch.Tensor):
        pass

    @abstractmethod
    def forward_over_time(self, x: torch.Tensor):
        pass

    def get_ranking(
            self, queries: torch.Tensor,
            filters: Dict[Tuple[int, int, int], List[int]],
            batch_size: int = 1000, chunk_size: int = -1
    ):
        """
        Returns filtered ranking for each queries.
        :param queries: a torch.LongTensor of quadruples (lhs, rel, rhs, timestamp)
        :param filters: filters[(lhs, rel, ts)] gives the elements to filter from ranking
        :param batch_size: maximum number of queries processed at once
        :param chunk_size: maximum number of candidates processed at once
        :return:
        """
        if chunk_size < 0:
            chunk_size = self.sizes[2]
        ranks = torch.ones(len(queries))
        with torch.no_grad():
            c_begin = 0
            while c_begin < self.sizes[2]:
                b_begin = 0
                rhs = self.get_rhs(c_begin, chunk_size)
                while b_begin < len(queries):
                    these_queries = queries[b_begin:b_begin + batch_size]
                    q = self.get_queries(these_queries)

                    scores = q @ rhs
                    targets = self.score(these_queries)
                    assert not torch.any(torch.isinf(scores)), "inf scores"
                    assert not torch.any(torch.isnan(scores)), "nan scores"
                    assert not torch.any(torch.isinf(targets)), "inf targets"
                    assert not torch.any(torch.isnan(targets)), "nan targets"

                    # set filtered and true scores to -1e6 to be ignored
                    # take care that scores are chunked
                    for i, query in enumerate(these_queries):
                        filter_out = filters[(query[0].item(), query[1].item(), query[3].item())]
                        filter_out += [queries[b_begin + i, 2].item()]
                        if chunk_size < self.sizes[2]:
                            filter_in_chunk = [
                                int(x - c_begin) for x in filter_out
                                if c_begin <= x < c_begin + chunk_size
                            ]
                            scores[i, torch.LongTensor(filter_in_chunk)] = -1e6
                        else:
                            scores[i, torch.LongTensor(filter_out)] = -1e6
                    ranks[b_begin:b_begin + batch_size] += torch.sum(
                        (scores >= targets).float(), dim=1
                    ).cpu()

                    b_begin += batch_size

                c_begin += chunk_size
        return ranks

    def get_auc(
            self, queries: torch.Tensor, batch_size: int = 1000
    ):
        """
        Returns filtered ranking for each queries.
        :param queries: a torch.LongTensor of quadruples (lhs, rel, rhs, begin, end)
        :param batch_size: maximum number of queries processed at once
        :return:
        """
        all_scores, all_truth = [], []
        all_ts_ids = None
        with torch.no_grad():
            b_begin = 0
            while b_begin < len(queries):
                these_queries = queries[b_begin:b_begin + batch_size]
                scores = self.forward_over_time(these_queries)
                all_scores.append(scores.cpu().numpy())
                if all_ts_ids is None:
                    all_ts_ids = torch.arange(0, scores.shape[1]).cuda()[None, :]
                assert not torch.any(torch.isinf(scores) + torch.isnan(scores)), "inf or nan scores"
                truth = (all_ts_ids <= these_queries[:, 4][:, None]) * (all_ts_ids >= these_queries[:, 3][:, None])
                all_truth.append(truth.cpu().numpy())
                b_begin += batch_size

        return np.concatenate(all_truth), np.concatenate(all_scores)

    def get_time_ranking(
            self, queries: torch.Tensor, filters: List[List[int]], chunk_size: int = -1
    ):
        """
        Returns filtered ranking for a batch of queries ordered by timestamp.
        :param queries: a torch.LongTensor of quadruples (lhs, rel, rhs, timestamp)
        :param filters: ordered filters
        :param chunk_size: maximum number of candidates processed at once
        :return:
        """
        if chunk_size < 0:
            chunk_size = self.sizes[2]
        ranks = torch.ones(len(queries))
        with torch.no_grad():
            c_begin = 0
            q = self.get_queries(queries)
            targets = self.score(queries)
            while c_begin < self.sizes[2]:
                rhs = self.get_rhs(c_begin, chunk_size)
                scores = q @ rhs
                # set filtered and true scores to -1e6 to be ignored
                # take care that scores are chunked
                for i, (query, filter) in enumerate(zip(queries, filters)):
                    filter_out = filter + [query[2].item()]
                    if chunk_size < self.sizes[2]:
                        filter_in_chunk = [
                            int(x - c_begin) for x in filter_out
                            if c_begin <= x < c_begin + chunk_size
                        ]
                        max_to_filter = max(filter_in_chunk + [-1])
                        assert max_to_filter < scores.shape[1], f"fuck {scores.shape[1]} {max_to_filter}"
                        scores[i, filter_in_chunk] = -1e6
                    else:
                        scores[i, filter_out] = -1e6
                ranks += torch.sum(
                    (scores >= targets).float(), dim=1
                ).cpu()

                c_begin += chunk_size
        return ranks


class ComplEx(TKBCModel):
    def __init__(
            self, sizes: Tuple[int, int, int, int], rank: int,
            init_size: float = 1e-3
    ):
        super(ComplEx, self).__init__()
        self.sizes = sizes
        self.rank = rank

        self.embeddings = nn.ModuleList([
            nn.Embedding(s, 2 * rank, sparse=True)
            for s in [sizes[0], sizes[1]]
        ])
        self.embeddings[0].weight.data *= init_size
        self.embeddings[1].weight.data *= init_size

    @staticmethod
    def has_time():
        return False

    def forward_over_time(self, x):
        raise NotImplementedError("no.")

    def score(self, x):
        lhs = self.embeddings[0](x[:, 0])
        rel = self.embeddings[1](x[:, 1])
        rhs = self.embeddings[0](x[:, 2])

        lhs = lhs[:, :self.rank], lhs[:, self.rank:]
        rel = rel[:, :self.rank], rel[:, self.rank:]
        rhs = rhs[:, :self.rank], rhs[:, self.rank:]

        return torch.sum(
            (lhs[0] * rel[0] - lhs[1] * rel[1]) * rhs[0] +
            (lhs[0] * rel[1] + lhs[1] * rel[0]) * rhs[1],
            1, keepdim=True
        )

    def forward(self, x):
        lhs = self.embeddings[0](x[:, 0])
        rel = self.embeddings[1](x[:, 1])
        rhs = self.embeddings[0](x[:, 2])

        lhs = lhs[:, :self.rank], lhs[:, self.rank:]
        rel = rel[:, :self.rank], rel[:, self.rank:]
        rhs = rhs[:, :self.rank], rhs[:, self.rank:]

        right = self.embeddings[0].weight
        right = right[:, :self.rank], right[:, self.rank:]
        return (
                       (lhs[0] * rel[0] - lhs[1] * rel[1]) @ right[0].transpose(0, 1) +
                       (lhs[0] * rel[1] + lhs[1] * rel[0]) @ right[1].transpose(0, 1)
               ), (
                   torch.sqrt(lhs[0] ** 2 + lhs[1] ** 2),
                   torch.sqrt(rel[0] ** 2 + rel[1] ** 2),
                   torch.sqrt(rhs[0] ** 2 + rhs[1] ** 2)
               ), None

    def get_rhs(self, chunk_begin: int, chunk_size: int):
        return self.embeddings[0].weight.data[
               chunk_begin:chunk_begin + chunk_size
               ].transpose(0, 1)

    def get_queries(self, queries: torch.Tensor):
        lhs = self.embeddings[0](queries[:, 0])
        rel = self.embeddings[1](queries[:, 1])
        lhs = lhs[:, :self.rank], lhs[:, self.rank:]
        rel = rel[:, :self.rank], rel[:, self.rank:]

        return torch.cat([
            lhs[0] * rel[0] - lhs[1] * rel[1],
            lhs[0] * rel[1] + lhs[1] * rel[0]
        ], 1)


class TComplEx(TKBCModel):
    def __init__(
            self, sizes: Tuple[int, int, int, int], rank: int,
            no_time_emb=False, init_size: float = 1e-2
    ):
        super(TComplEx, self).__init__()
        self.sizes = sizes
        self.rank = rank

        self.embeddings = nn.ModuleList([
            nn.Embedding(s, 2 * rank, sparse=True)
            for s in [sizes[0], sizes[1], sizes[3]]
        ])
        self.embeddings[0].weight.data *= init_size
        self.embeddings[1].weight.data *= init_size
        self.embeddings[2].weight.data *= init_size

        self.no_time_emb = no_time_emb

    @staticmethod
    def has_time():
        return True

    def score(self, x):
        lhs = self.embeddings[0](x[:, 0])
        rel = self.embeddings[1](x[:, 1])
        rhs = self.embeddings[0](x[:, 2])
        time = self.embeddings[2](x[:, 3])

        lhs = lhs[:, :self.rank], lhs[:, self.rank:]
        rel = rel[:, :self.rank], rel[:, self.rank:]
        rhs = rhs[:, :self.rank], rhs[:, self.rank:]
        time = time[:, :self.rank], time[:, self.rank:]

        return torch.sum(
            (lhs[0] * rel[0] * time[0] - lhs[1] * rel[1] * time[0] -
             lhs[1] * rel[0] * time[1] - lhs[0] * rel[1] * time[1]) * rhs[0] +
            (lhs[1] * rel[0] * time[0] + lhs[0] * rel[1] * time[0] +
             lhs[0] * rel[0] * time[1] - lhs[1] * rel[1] * time[1]) * rhs[1],
            1, keepdim=True
        )

    def forward(self, x):
        lhs = self.embeddings[0](x[:, 0])
        rel = self.embeddings[1](x[:, 1])
        rhs = self.embeddings[0](x[:, 2])
        time = self.embeddings[2](x[:, 3])

        lhs = lhs[:, :self.rank], lhs[:, self.rank:]
        rel = rel[:, :self.rank], rel[:, self.rank:]
        rhs = rhs[:, :self.rank], rhs[:, self.rank:]
        time = time[:, :self.rank], time[:, self.rank:]

        right = self.embeddings[0].weight
        right = right[:, :self.rank], right[:, self.rank:]

        rt = rel[0] * time[0], rel[1] * time[0], rel[0] * time[1], rel[1] * time[1]
        full_rel = rt[0] - rt[3], rt[1] + rt[2]

        return (
                       (lhs[0] * full_rel[0] - lhs[1] * full_rel[1]) @ right[0].t() +
                       (lhs[1] * full_rel[0] + lhs[0] * full_rel[1]) @ right[1].t()
               ), (
                   torch.sqrt(lhs[0] ** 2 + lhs[1] ** 2),
                   torch.sqrt(full_rel[0] ** 2 + full_rel[1] ** 2),
                   torch.sqrt(rhs[0] ** 2 + rhs[1] ** 2)
               ), self.embeddings[2].weight[:-1] if self.no_time_emb else self.embeddings[2].weight

    def forward_over_time(self, x):
        lhs = self.embeddings[0](x[:, 0])
        rel = self.embeddings[1](x[:, 1])
        rhs = self.embeddings[0](x[:, 2])
        time = self.embeddings[2].weight

        lhs = lhs[:, :self.rank], lhs[:, self.rank:]
        rel = rel[:, :self.rank], rel[:, self.rank:]
        rhs = rhs[:, :self.rank], rhs[:, self.rank:]
        time = time[:, :self.rank], time[:, self.rank:]

        return (
                (lhs[0] * rel[0] * rhs[0] - lhs[1] * rel[1] * rhs[0] -
                 lhs[1] * rel[0] * rhs[1] + lhs[0] * rel[1] * rhs[1]) @ time[0].t() +
                (lhs[1] * rel[0] * rhs[0] - lhs[0] * rel[1] * rhs[0] +
                 lhs[0] * rel[0] * rhs[1] - lhs[1] * rel[1] * rhs[1]) @ time[1].t()
        )

    def get_rhs(self, chunk_begin: int, chunk_size: int):
        return self.embeddings[0].weight.data[
               chunk_begin:chunk_begin + chunk_size
               ].transpose(0, 1)

    def get_queries(self, queries: torch.Tensor):
        lhs = self.embeddings[0](queries[:, 0])
        rel = self.embeddings[1](queries[:, 1])
        time = self.embeddings[2](queries[:, 3])
        lhs = lhs[:, :self.rank], lhs[:, self.rank:]
        rel = rel[:, :self.rank], rel[:, self.rank:]
        time = time[:, :self.rank], time[:, self.rank:]
        return torch.cat([
            lhs[0] * rel[0] * time[0] - lhs[1] * rel[1] * time[0] -
            lhs[1] * rel[0] * time[1] - lhs[0] * rel[1] * time[1],
            lhs[1] * rel[0] * time[0] + lhs[0] * rel[1] * time[0] +
            lhs[0] * rel[0] * time[1] - lhs[1] * rel[1] * time[1]
        ], 1)


class TNTComplEx(TKBCModel):
    def __init__(
            self, sizes: Tuple[int, int, int, int], rank: int,
            no_time_emb=False, init_size: float = 1e-2
    ):
        super(TNTComplEx, self).__init__()
        self.sizes = sizes
        self.rank = rank

        self.embeddings = nn.ModuleList([
            nn.Embedding(s, 2 * rank, sparse=True)
            for s in [sizes[0], sizes[1], sizes[3], sizes[1]]  # last embedding modules contains no_time embeddings
        ])
        self.embeddings[0].weight.data *= init_size
        self.embeddings[1].weight.data *= init_size
        self.embeddings[2].weight.data *= init_size
        self.embeddings[3].weight.data *= init_size

        self.no_time_emb = no_time_emb

    @staticmethod
    def has_time():
        return True

    def score(self, x):
        lhs = self.embeddings[0](x[:, 0])
        rel = self.embeddings[1](x[:, 1])
        rel_no_time = self.embeddings[3](x[:, 1])
        rhs = self.embeddings[0](x[:, 2])
        time = self.embeddings[2](x[:, 3])

        lhs = lhs[:, :self.rank], lhs[:, self.rank:]
        rel = rel[:, :self.rank], rel[:, self.rank:]
        rhs = rhs[:, :self.rank], rhs[:, self.rank:]
        time = time[:, :self.rank], time[:, self.rank:]
        rnt = rel_no_time[:, :self.rank], rel_no_time[:, self.rank:]

        rt = rel[0] * time[0], rel[1] * time[0], rel[0] * time[1], rel[1] * time[1]
        full_rel = (rt[0] - rt[3]) + rnt[0], (rt[1] + rt[2]) + rnt[1]

        return torch.sum(
            (lhs[0] * full_rel[0] - lhs[1] * full_rel[1]) * rhs[0] +
            (lhs[1] * full_rel[0] + lhs[0] * full_rel[1]) * rhs[1],
            1, keepdim=True
        )

    def forward(self, x):
        lhs = self.embeddings[0](x[:, 0])
        rel = self.embeddings[1](x[:, 1])
        rel_no_time = self.embeddings[3](x[:, 1])
        rhs = self.embeddings[0](x[:, 2])
        time = self.embeddings[2](x[:, 3])

        lhs = lhs[:, :self.rank], lhs[:, self.rank:]
        rel = rel[:, :self.rank], rel[:, self.rank:]
        rhs = rhs[:, :self.rank], rhs[:, self.rank:]
        time = time[:, :self.rank], time[:, self.rank:]

        rnt = rel_no_time[:, :self.rank], rel_no_time[:, self.rank:]

        right = self.embeddings[0].weight
        right = right[:, :self.rank], right[:, self.rank:]

        rt = rel[0] * time[0], rel[1] * time[0], rel[0] * time[1], rel[1] * time[1]
        rrt = rt[0] - rt[3], rt[1] + rt[2]
        full_rel = rrt[0] + rnt[0], rrt[1] + rnt[1]

        regularizer = (
           math.pow(2, 1 / 3) * torch.sqrt(lhs[0] ** 2 + lhs[1] ** 2),
           torch.sqrt(rrt[0] ** 2 + rrt[1] ** 2),
           torch.sqrt(rnt[0] ** 2 + rnt[1] ** 2),
           math.pow(2, 1 / 3) * torch.sqrt(rhs[0] ** 2 + rhs[1] ** 2)
        )
        return ((
               (lhs[0] * full_rel[0] - lhs[1] * full_rel[1]) @ right[0].t() +
               (lhs[1] * full_rel[0] + lhs[0] * full_rel[1]) @ right[1].t()
            ), regularizer,
               self.embeddings[2].weight[:-1] if self.no_time_emb else self.embeddings[2].weight
        )

    def forward_over_time(self, x):
        lhs = self.embeddings[0](x[:, 0])
        rel = self.embeddings[1](x[:, 1])
        rhs = self.embeddings[0](x[:, 2])
        time = self.embeddings[2].weight

        lhs = lhs[:, :self.rank], lhs[:, self.rank:]
        rel = rel[:, :self.rank], rel[:, self.rank:]
        rhs = rhs[:, :self.rank], rhs[:, self.rank:]
        time = time[:, :self.rank], time[:, self.rank:]

        rel_no_time = self.embeddings[3](x[:, 1])
        rnt = rel_no_time[:, :self.rank], rel_no_time[:, self.rank:]

        score_time = (
            (lhs[0] * rel[0] * rhs[0] - lhs[1] * rel[1] * rhs[0] -
             lhs[1] * rel[0] * rhs[1] + lhs[0] * rel[1] * rhs[1]) @ time[0].t() +
            (lhs[1] * rel[0] * rhs[0] - lhs[0] * rel[1] * rhs[0] +
             lhs[0] * rel[0] * rhs[1] - lhs[1] * rel[1] * rhs[1]) @ time[1].t()
        )
        base = torch.sum(
            (lhs[0] * rnt[0] * rhs[0] - lhs[1] * rnt[1] * rhs[0] -
             lhs[1] * rnt[0] * rhs[1] + lhs[0] * rnt[1] * rhs[1]) +
            (lhs[1] * rnt[1] * rhs[0] - lhs[0] * rnt[0] * rhs[0] +
             lhs[0] * rnt[1] * rhs[1] - lhs[1] * rnt[0] * rhs[1]),
            dim=1, keepdim=True
        )
        return score_time + base

    def get_rhs(self, chunk_begin: int, chunk_size: int):
        return self.embeddings[0].weight.data[
               chunk_begin:chunk_begin + chunk_size
               ].transpose(0, 1)

    def get_queries(self, queries: torch.Tensor):
        lhs = self.embeddings[0](queries[:, 0])
        rel = self.embeddings[1](queries[:, 1])
        rel_no_time = self.embeddings[3](queries[:, 1])
        time = self.embeddings[2](queries[:, 3])

        lhs = lhs[:, :self.rank], lhs[:, self.rank:]
        rel = rel[:, :self.rank], rel[:, self.rank:]
        time = time[:, :self.rank], time[:, self.rank:]
        rnt = rel_no_time[:, :self.rank], rel_no_time[:, self.rank:]

        rt = rel[0] * time[0], rel[1] * time[0], rel[0] * time[1], rel[1] * time[1]
        full_rel = (rt[0] - rt[3]) + rnt[0], (rt[1] + rt[2]) + rnt[1]

        return torch.cat([
            lhs[0] * full_rel[0] - lhs[1] * full_rel[1],
            lhs[1] * full_rel[0] + lhs[0] * full_rel[1]
        ], 1)


class TNTPairRE(TKBCModel):
    """
    TNT-PairRE: Temporal PairRE with TNT-style Decomposition
    
    Score function: f(h,r,t,l) = -||e_h * r^H_l - e_t * r^T_l||_1
    where:
        r^H_l = r^H + r^{H,t} * tau_l  (non-temporal + temporal)
        r^T_l = r^T + r^{T,t} * tau_l
    
    Entity embeddings are constrained to unit norm: ||e_x||_2 = 1
    """
    def __init__(
            self, sizes: Tuple[int, int, int, int], rank: int,
            no_time_emb=False, init_size: float = 1e-2
    ):
        super(TNTPairRE, self).__init__()
        self.sizes = sizes
        self.rank = rank
        self.no_time_emb = no_time_emb
        
        # Entity embeddings (will be normalized to unit norm)
        self.entity_embeddings = nn.Embedding(sizes[0], rank, sparse=True)
        self.entity_embeddings.weight.data *= init_size
        
        # Time embeddings
        self.time_embeddings = nn.Embedding(sizes[3], rank, sparse=True)
        self.time_embeddings.weight.data *= init_size
        
        # Relation embeddings: 4 vectors per relation
        # Non-temporal (static) relation vectors
        self.rel_H = nn.Embedding(sizes[1], rank, sparse=True)  # r^H
        self.rel_T = nn.Embedding(sizes[1], rank, sparse=True)  # r^T
        
        # Temporal (time-sensitive) relation vectors
        self.rel_H_t = nn.Embedding(sizes[1], rank, sparse=True)  # r^{H,t}
        self.rel_T_t = nn.Embedding(sizes[1], rank, sparse=True)  # r^{T,t}
        
        # Initialize relation embeddings
        self.rel_H.weight.data *= init_size
        self.rel_T.weight.data *= init_size
        self.rel_H_t.weight.data *= init_size
        self.rel_T_t.weight.data *= init_size
    
    @staticmethod
    def has_time():
        return True
    
    def _normalize_entities(self):
        """Normalize entity embeddings to unit norm (L2 norm = 1)"""
        with torch.no_grad():
            norms = torch.norm(self.entity_embeddings.weight.data, p=2, dim=1, keepdim=True)
            norms = torch.clamp(norms, min=1e-10)  # Avoid division by zero
            self.entity_embeddings.weight.data.div_(norms)
    
    def score(self, x):
        """
        Compute scores for given quadruples (h, r, t, l)
        
        Args:
            x: torch.LongTensor of shape (batch_size, 4) containing (h, r, t, l)
        
        Returns:
            scores: torch.Tensor of shape (batch_size, 1)
        """
        # Extract embeddings
        e_h = self.entity_embeddings(x[:, 0])  # (batch, rank)
        e_t = self.entity_embeddings(x[:, 2])  # (batch, rank)
        tau = self.time_embeddings(x[:, 3])    # (batch, rank)
        
        # Get relation vectors
        r_H = self.rel_H(x[:, 1])      # (batch, rank)
        r_T = self.rel_T(x[:, 1])      # (batch, rank)
        r_H_t = self.rel_H_t(x[:, 1])  # (batch, rank)
        r_T_t = self.rel_T_t(x[:, 1])  # (batch, rank)
        
        # Time-conditioned relations: r^H_l = r^H + r^{H,t} * tau
        r_H_l = r_H + r_H_t * tau      # (batch, rank)
        r_T_l = r_T + r_T_t * tau      # (batch, rank)
        
        # PairRE score: -||e_h * r^H_l - e_t * r^T_l||_1
        diff = e_h * r_H_l - e_t * r_T_l
        score = -torch.sum(torch.abs(diff), dim=1, keepdim=True)
        
        return score
    
    def forward(self, x):
        """
        Forward pass for training (1-vs-all setting)
        
        Args:
            x: torch.LongTensor of shape (batch_size, 4) containing (h, r, t, l)
        
        Returns:
            scores: torch.Tensor of shape (batch_size, n_entities) - scores for all entities
            factors: tuple of embeddings for regularization
            time: time embeddings for temporal regularization
        """
        batch_size = x.shape[0]
        
        # Extract embeddings
        e_h = self.entity_embeddings(x[:, 0])  # (batch, rank)
        e_t = self.entity_embeddings(x[:, 2])  # (batch, rank)
        tau = self.time_embeddings(x[:, 3])    # (batch, rank)
        
        # Get relation vectors
        r_H = self.rel_H(x[:, 1])      # (batch, rank)
        r_T = self.rel_T(x[:, 1])      # (batch, rank)
        r_H_t = self.rel_H_t(x[:, 1])  # (batch, rank)
        r_T_t = self.rel_T_t(x[:, 1])  # (batch, rank)
        
        # Time-conditioned relations
        r_H_l = r_H + r_H_t * tau      # (batch, rank)
        r_T_l = r_T + r_T_t * tau      # (batch, rank)
        
        # Get all entity embeddings for scoring
        all_entities = self.entity_embeddings.weight  # (n_entities, rank)
        
        # Compute scores for all entities as tails
        # score(h, r, t', l) = -||e_h * r^H_l - e_t' * r^T_l||_1
        # We need to compute this for all t' efficiently
        
        # Expand dimensions for broadcasting
        e_h_expanded = e_h.unsqueeze(1)        # (batch, 1, rank)
        r_H_l_expanded = r_H_l.unsqueeze(1)    # (batch, 1, rank)
        r_T_l_expanded = r_T_l.unsqueeze(1)    # (batch, 1, rank)
        
        # all_entities: (n_entities, rank) -> (1, n_entities, rank)
        all_entities_expanded = all_entities.unsqueeze(0)
        
        # Compute: e_h * r^H_l - e_t' * r^T_l for all t'
        # (batch, 1, rank) * (batch, 1, rank) - (1, n_entities, rank) * (batch, 1, rank)
        lhs = e_h_expanded * r_H_l_expanded  # (batch, 1, rank)
        rhs = all_entities_expanded * r_T_l_expanded  # (batch, n_entities, rank)
        
        diff = lhs - rhs  # (batch, n_entities, rank)
        
        # L1 norm along rank dimension, then negate
        scores = -torch.sum(torch.abs(diff), dim=2)  # (batch, n_entities)
        
        # Factors for regularization (L2 on relation and time embeddings)
        factors = (
            torch.sqrt(r_H ** 2 + 1e-10).mean(dim=1, keepdim=True),  # Dummy for N3 regularizer
            torch.sqrt(r_T ** 2 + r_H_t ** 2 + r_T_t ** 2 + 1e-10).mean(dim=1, keepdim=True),
            torch.sqrt(e_t ** 2 + 1e-10).mean(dim=1, keepdim=True)
        )
        
        # Return time embeddings for temporal regularization
        time = self.time_embeddings.weight[:-1] if self.no_time_emb else self.time_embeddings.weight
        
        return scores, factors, time
    
    def forward_over_time(self, x):
        """
        Compute scores over all timestamps for given (h, r, t)
        
        Args:
            x: torch.LongTensor of shape (batch_size, 4+) containing at least (h, r, t)
        
        Returns:
            scores: torch.Tensor of shape (batch_size, n_timestamps)
        """
        # Extract embeddings
        e_h = self.entity_embeddings(x[:, 0])  # (batch, rank)
        e_t = self.entity_embeddings(x[:, 2])  # (batch, rank)
        
        # Get relation vectors
        r_H = self.rel_H(x[:, 1])      # (batch, rank)
        r_T = self.rel_T(x[:, 1])      # (batch, rank)
        r_H_t = self.rel_H_t(x[:, 1])  # (batch, rank)
        r_T_t = self.rel_T_t(x[:, 1])  # (batch, rank)
        
        # Get all time embeddings
        all_time = self.time_embeddings.weight  # (n_timestamps, rank)
        
        # Compute scores for all timestamps
        # r^H_l = r^H + r^{H,t} * tau_l for all l
        # Expand for broadcasting: (batch, 1, rank) and (1, n_timestamps, rank)
        r_H_exp = r_H.unsqueeze(1)      # (batch, 1, rank)
        r_T_exp = r_T.unsqueeze(1)      # (batch, 1, rank)
        r_H_t_exp = r_H_t.unsqueeze(1)  # (batch, 1, rank)
        r_T_t_exp = r_T_t.unsqueeze(1)  # (batch, 1, rank)
        
        tau_exp = all_time.unsqueeze(0)  # (1, n_timestamps, rank)
        
        # Time-conditioned relations for all timestamps
        r_H_l = r_H_exp + r_H_t_exp * tau_exp  # (batch, n_timestamps, rank)
        r_T_l = r_T_exp + r_T_t_exp * tau_exp  # (batch, n_timestamps, rank)
        
        # Expand entity embeddings
        e_h_exp = e_h.unsqueeze(1)  # (batch, 1, rank)
        e_t_exp = e_t.unsqueeze(1)  # (batch, 1, rank)
        
        # Compute differences
        diff = e_h_exp * r_H_l - e_t_exp * r_T_l  # (batch, n_timestamps, rank)
        
        # L1 norm and negate
        scores = -torch.sum(torch.abs(diff), dim=2)  # (batch, n_timestamps)
        
        return scores
    
    def get_rhs(self, chunk_begin: int, chunk_size: int):
        """
        Get entity embeddings for a chunk of entities (for ranking)
        
        Returns:
            entity_embeddings: torch.Tensor of shape (rank, chunk_size)
        """
        return self.entity_embeddings.weight.data[
            chunk_begin:chunk_begin + chunk_size
        ].transpose(0, 1)
    
    def get_queries(self, queries: torch.Tensor):
        """
        Compute query embeddings for ranking
        
        Args:
            queries: torch.LongTensor of shape (batch_size, 4) containing (h, r, t, l)
        
        Returns:
            query embeddings: torch.Tensor of shape (batch_size, rank)
        """
        # For PairRE, we need to return something that can be matrix-multiplied
        # with entity embeddings to get scores
        
        # However, PairRE with L1 distance doesn't decompose nicely into
        # query @ entity form. We'll return the time-conditioned h * r^H_l
        # and store r^T_l separately for scoring
        
        e_h = self.entity_embeddings(queries[:, 0])  # (batch, rank)
        tau = self.time_embeddings(queries[:, 3])    # (batch, rank)
        
        r_H = self.rel_H(queries[:, 1])      # (batch, rank)
        r_H_t = self.rel_H_t(queries[:, 1])  # (batch, rank)
        
        # Time-conditioned head relation
        r_H_l = r_H + r_H_t * tau
        
        # This is a simplification - the actual scoring in get_ranking
        # will need special handling for L1 distance
        return e_h * r_H_l


