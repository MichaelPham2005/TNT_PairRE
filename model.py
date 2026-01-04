#!/usr/bin/python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class BaselinePairRE(nn.Module):
    """
    Baseline PairRE model WITHOUT temporal modeling.
    Treats ICEWS14 as static KG - ignores timestamps during training.
    """
    
    def __init__(self, model_name, nentity, nrelation, hidden_dim, gamma,
                 double_entity_embedding=False, double_relation_embedding=False):
        super(BaselinePairRE, self).__init__()
        self.model_name = model_name
        self.nentity = nentity
        self.nrelation = nrelation
        self.hidden_dim = hidden_dim
        self.epsilon = 2.0
        
        self.gamma = nn.Parameter(
            torch.Tensor([gamma]), 
            requires_grad=False
        )
        
        self.embedding_range = nn.Parameter(
            torch.Tensor([(self.gamma.item() + self.epsilon) / hidden_dim]), 
            requires_grad=False
        )
        
        # Entity embeddings (static)
        self.entity_embedding = nn.Parameter(torch.zeros(nentity, hidden_dim))
        nn.init.uniform_(
            tensor=self.entity_embedding, 
            a=-self.embedding_range.item(), 
            b=self.embedding_range.item()
        )
        
        # Relation embedding (double for head and tail transformations)
        if double_relation_embedding:
            self.relation_embedding = nn.Parameter(torch.zeros(nrelation, hidden_dim * 2))
        else:
            self.relation_embedding = nn.Parameter(torch.zeros(nrelation, hidden_dim))
        nn.init.uniform_(
            tensor=self.relation_embedding, 
            a=-self.embedding_range.item(), 
            b=self.embedding_range.item()
        )
        
    def forward(self, sample, mode='single'):
        """
        Forward pass - PairRE scoring WITHOUT timestamps.
        
        Args:
            sample: tuple of (head, relation, tail, timestamps)
                    BUT we ignore timestamps!
            mode: 'single', 'head-batch', or 'tail-batch'
        
        Returns:
            score: [batch_size] or [batch_size, negative_sample_size]
        """
        # Unpack sample - timestamps ignored
        if len(sample) == 4:
            head, relation, tail, timestamps = sample
            # timestamps not used in baseline!
        else:
            raise ValueError('Sample must be (head, relation, tail, timestamp)')
        
        if mode == 'single':
            batch_size, negative_sample_size = head.size(0), 1
            
            head_emb = torch.index_select(
                self.entity_embedding, 
                dim=0, 
                index=head
            ).unsqueeze(1)  # [batch, 1, hidden_dim]
            
            tail_emb = torch.index_select(
                self.entity_embedding, 
                dim=0, 
                index=tail
            ).unsqueeze(1)  # [batch, 1, hidden_dim]
            
        elif mode == 'head-batch':
            tail_part, head_part = head.size(0), tail.size(0)
            batch_size, negative_sample_size = tail_part, head_part
            
            head_emb = self.entity_embedding.unsqueeze(0)  # [1, nentity, hidden_dim]
            
            tail_emb = torch.index_select(
                self.entity_embedding, 
                dim=0, 
                index=tail
            ).unsqueeze(1)  # [tail_part, 1, hidden_dim]
            
        elif mode == 'tail-batch':
            head_part, tail_part = head.size(0), tail.size(0)
            batch_size, negative_sample_size = head_part, tail_part
            
            head_emb = torch.index_select(
                self.entity_embedding, 
                dim=0,
                index=head
            ).unsqueeze(1)  # [head_part, 1, hidden_dim]
            
            tail_emb = self.entity_embedding.unsqueeze(0)  # [1, nentity, hidden_dim]
        else:
            raise ValueError('mode %s not supported' % mode)
        
        # Get relation embedding
        relation_emb = torch.index_select(
            self.relation_embedding,
            dim=0,
            index=relation
        ).unsqueeze(1)  # [batch, 1, hidden_dim or hidden_dim*2]
        
        # PairRE score
        score = self.pairre_score(head_emb, relation_emb, tail_emb)
        
        return score
    
    def pairre_score(self, head, relation, tail):
        """
        Standard PairRE scoring function.
        score = gamma - ||head ⊙ r_h - tail ⊙ r_t||_1
        """
        # Get head and tail relation parts
        if relation.size(-1) == self.hidden_dim * 2:
            r_h = relation[:, :, :self.hidden_dim]  # [batch, 1, hidden_dim]
            r_t = relation[:, :, self.hidden_dim:]  # [batch, 1, hidden_dim]
        else:
            r_h = relation
            r_t = relation
        
        # PairRE score: gamma - L1(head * r_h - tail * r_t)
        score = self.gamma - torch.norm(head * r_h - tail * r_t, p=1, dim=-1)
        
        return score
    
    @staticmethod
    def train_step(model, optimizer, train_iterator, args):
        """
        Training step with contrastive loss.
        Identical to temporal model but ignores timestamps.
        """
        model.train()
        optimizer.zero_grad()
        
        positive_sample, negative_sample, subsampling_weight, mode, timestamps = next(train_iterator)
        
        if args.cuda:
            positive_sample = positive_sample.cuda()
            negative_sample = negative_sample.cuda()
            subsampling_weight = subsampling_weight.cuda()
            timestamps = timestamps.cuda()
        
        # Parse positive sample
        head_pos = positive_sample[:, 0]
        relation_pos = positive_sample[:, 1]
        tail_pos = positive_sample[:, 2]
        
        # Positive score (timestamps passed but ignored in forward)
        positive_score = model((head_pos, relation_pos, tail_pos, timestamps))
        
        # Negative scores
        if mode == 'head-batch':
            negative_score = model((negative_sample, relation_pos, tail_pos, timestamps), mode)
        elif mode == 'tail-batch':
            negative_score = model((head_pos, relation_pos, negative_sample, timestamps), mode)
        else:
            raise ValueError('Training batch mode %s not supported' % mode)
        
        # Self-adversarial negative sampling
        if args.negative_adversarial_sampling:
            negative_score = (F.softmax(negative_score * args.adversarial_temperature, dim=1).detach() 
                            * F.logsigmoid(-negative_score)).sum(dim=1)
        else:
            negative_score = F.logsigmoid(-negative_score).mean(dim=1)
        
        positive_score = F.logsigmoid(positive_score).squeeze(dim=1)
        
        positive_sample_loss = - (subsampling_weight * positive_score).sum() / subsampling_weight.sum()
        negative_sample_loss = - (subsampling_weight * negative_score).sum() / subsampling_weight.sum()
        
        loss = (positive_sample_loss + negative_sample_loss) / 2
        
        # Regularization
        if args.regularization != 0.0:
            # Handle DataParallel wrapper
            actual_model = model.module if hasattr(model, 'module') else model
            regularization = args.regularization * (
                actual_model.entity_embedding.norm(p=3) ** 3 + 
                actual_model.relation_embedding.norm(p=3) ** 3
            )
            loss = loss + regularization
        
        loss.backward()
        optimizer.step()
        
        log = {
            'positive_sample_loss': positive_sample_loss.item(),
            'negative_sample_loss': negative_sample_loss.item(),
            'loss': loss.item()
        }
        
        return log
    
    @staticmethod
    def test_step(model, test_triples, all_true_triples, args):
        """
        Evaluation step - SAME as temporal model (temporal filtering).
        This ensures fair comparison!
        """
        from dataloader import TemporalTestDataset
        from torch.utils.data import DataLoader
        
        model.eval()
        
        test_dataloader_head = DataLoader(
            TemporalTestDataset(
                test_triples, 
                all_true_triples,
                args.nentity,
                args.nrelation,
                'head-batch'
            ), 
            batch_size=args.test_batch_size,
            num_workers=max(1, args.cpu_num//2), 
            collate_fn=TemporalTestDataset.collate_fn
        )
        
        test_dataloader_tail = DataLoader(
            TemporalTestDataset(
                test_triples,
                all_true_triples,
                args.nentity,
                args.nrelation,
                'tail-batch'
            ), 
            batch_size=args.test_batch_size,
            num_workers=max(1, args.cpu_num//2), 
            collate_fn=TemporalTestDataset.collate_fn
        )
        
        test_dataset_list = [test_dataloader_head, test_dataloader_tail]
        
        logs = []
        
        step = 0
        total_steps = sum([len(dataset) for dataset in test_dataset_list])
        
        with torch.no_grad():
            for test_dataset in test_dataset_list:
                for positive_sample, negative_sample, filter_bias, mode, timestamps in test_dataset:
                    if args.cuda:
                        positive_sample = positive_sample.cuda()
                        negative_sample = negative_sample.cuda()
                        filter_bias = filter_bias.cuda()
                        timestamps = timestamps.cuda()
                    
                    batch_size = positive_sample.size(0)
                    
                    # Construct samples for scoring
                    if mode == 'head-batch':
                        head = negative_sample  # [batch, nentity]
                        relation = positive_sample[:, 1]  # [batch]
                        tail = positive_sample[:, 2]  # [batch]
                    elif mode == 'tail-batch':
                        head = positive_sample[:, 0]  # [batch]
                        relation = positive_sample[:, 1]  # [batch]
                        tail = negative_sample  # [batch, nentity]
                    
                    # Call model forward (timestamps ignored)
                    if hasattr(model, 'module'):
                        score = model.module.forward((head, relation, tail, timestamps), mode)
                    else:
                        score = model((head, relation, tail, timestamps), mode)
                    score += filter_bias
                    
                    argsort = torch.argsort(score, dim=1, descending=True)
                    
                    if mode == 'head-batch':
                        positive_arg = positive_sample[:, 0]
                    elif mode == 'tail-batch':
                        positive_arg = positive_sample[:, 2]
                    else:
                        raise ValueError('mode %s not supported' % mode)
                    
                    for i in range(batch_size):
                        ranking = (argsort[i, :] == positive_arg[i]).nonzero()
                        assert ranking.size(0) == 1
                        
                        ranking = 1 + ranking.item()
                        logs.append({
                            'MRR': 1.0/ranking,
                            'MR': float(ranking),
                            'HITS@1': 1.0 if ranking <= 1 else 0.0,
                            'HITS@3': 1.0 if ranking <= 3 else 0.0,
                            'HITS@10': 1.0 if ranking <= 10 else 0.0,
                        })
                    
                    if step % args.test_log_steps == 0:
                        logging.info('Evaluating the model... (%d/%d)' % (step, total_steps))
                    
                    step += 1
        
        metrics = {}
        for metric in logs[0].keys():
            metrics[metric] = sum([log[metric] for log in logs])/len(logs)
        
        return metrics
