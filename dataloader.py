#!/usr/bin/python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
import pickle
from torch.utils.data import Dataset


class TemporalTrainDataset(Dataset):
    """
    Dataset for temporal knowledge graph quadruples: (head, relation, tail, timestamp)
    
    Supports negative sampling with both uniform and self-adversarial strategies
    """
    
    def __init__(self, triples, nentity, nrelation, negative_sample_size, mode):
        """
        Args:
            triples: List of (head_id, relation_id, tail_id, timestamp, original_ts) tuples
            nentity: Number of entities
            nrelation: Number of relations
            negative_sample_size: Number of negative samples per positive
            mode: 'head-batch' or 'tail-batch' for negative sampling
        """
        self.len = len(triples)
        self.triples = triples
        self.nentity = nentity
        self.nrelation = nrelation
        self.negative_sample_size = negative_sample_size
        self.mode = mode
        
        # Build true triple set for filtering (optional - can be used for filtered metrics)
        self.true_head = {}
        self.true_tail = {}
        
        for sample in triples:
            if len(sample) == 4:
                head, relation, tail, timestamp = sample
            else:
                head, relation, tail, timestamp, _ = sample
            
            if (head, relation) not in self.true_tail:
                self.true_tail[(head, relation)] = []
            self.true_tail[(head, relation)].append(tail)
            
            if (relation, tail) not in self.true_head:
                self.true_head[(relation, tail)] = []
            self.true_head[(relation, tail)].append(head)
        
        # Convert to sets for efficient lookup
        for key in self.true_head:
            self.true_head[key] = set(self.true_head[key])
        for key in self.true_tail:
            self.true_tail[key] = set(self.true_tail[key])
    
    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        """
        Returns:
            positive_sample: [head, relation, tail]
            negative_sample: [negative_sample_size] of entity IDs
            subsample_weight: float - frequency-based weight
            mode: 'head-batch' or 'tail-batch'
            timestamp: float - normalized timestamp in [0, 1]
        """
        positive_sample = self.triples[idx]
        if len(positive_sample) == 4:
            head, relation, tail, timestamp = positive_sample
        else:
            head, relation, tail, timestamp, _ = positive_sample
        
        # Create positive sample as tensor
        positive_sample_tensor = torch.LongTensor([head, relation, tail])
        timestamp_tensor = torch.FloatTensor([timestamp])
        
        # Compute subsampling weight (based on frequency)
        # Higher weight for rare entities
        subsampling_weight = self.count[(head, relation)] + self.count[(tail, -relation-1)]
        subsampling_weight = torch.sqrt(1 / torch.Tensor([subsampling_weight]))
        
        # Generate negative samples
        negative_sample_list = []
        negative_sample_size = 0
        
        while negative_sample_size < self.negative_sample_size:
            # Sample random entities
            negative_sample_batch = np.random.randint(self.nentity, size=self.negative_sample_size * 2)
            
            # Filter out true positives (optional - comment out for faster training)
            if self.mode == 'head-batch':
                mask = np.in1d(
                    negative_sample_batch,
                    self.true_head[(relation, tail)],
                    assume_unique=True,
                    invert=True
                )
            elif self.mode == 'tail-batch':
                mask = np.in1d(
                    negative_sample_batch,
                    self.true_tail[(head, relation)],
                    assume_unique=True,
                    invert=True
                )
            else:
                raise ValueError('Training batch mode %s not supported' % self.mode)
            
            negative_sample_batch = negative_sample_batch[mask]
            negative_sample_list.append(negative_sample_batch)
            negative_sample_size += negative_sample_batch.size
        
        negative_sample = np.concatenate(negative_sample_list)[:self.negative_sample_size]
        negative_sample = torch.from_numpy(negative_sample)
        
        return positive_sample_tensor, negative_sample, subsampling_weight, self.mode, timestamp_tensor
    
    @staticmethod
    def collate_fn(data):
        """
        Custom collate function for batching
        """
        positive_sample = torch.stack([_[0] for _ in data], dim=0)
        negative_sample = torch.stack([_[1] for _ in data], dim=0)
        subsample_weight = torch.cat([_[2] for _ in data], dim=0)
        mode = data[0][3]
        timestamps = torch.cat([_[4] for _ in data], dim=0)
        
        return positive_sample, negative_sample, subsample_weight, mode, timestamps


class TemporalTestDataset(Dataset):
    """
    Dataset for temporal KG evaluation (ranking task)
    """
    
    def __init__(self, triples, all_true_triples, nentity, nrelation, mode):
        """
        Args:
            triples: Test triples to evaluate
            all_true_triples: All triples (train + valid + test) for filtering
            nentity: Number of entities
            nrelation: Number of relations
            mode: 'head-batch' or 'tail-batch'
        """
        self.len = len(triples)
        self.triple_set = set(all_true_triples)
        self.triples = triples
        self.nentity = nentity
        self.nrelation = nrelation
        self.mode = mode
    
    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        sample = self.triples[idx]
        if len(sample) == 4:
            head, relation, tail, timestamp = sample
        else:
            head, relation, tail, timestamp, _ = sample
        
        if self.mode == 'head-batch':
            tmp = [(0, rand_head) if (rand_head, relation, tail, timestamp) not in self.triple_set
                   else (-1, head) for rand_head in range(self.nentity)]
            tmp[head] = (0, head)
        elif self.mode == 'tail-batch':
            tmp = [(0, rand_tail) if (head, relation, rand_tail, timestamp) not in self.triple_set
                   else (-1, tail) for rand_tail in range(self.nentity)]
            tmp[tail] = (0, tail)
        else:
            raise ValueError('Negative batch mode %s not supported' % self.mode)
        
        tmp = torch.LongTensor(tmp)
        filter_bias = tmp[:, 0].float()
        negative_sample = tmp[:, 1]
        
        positive_sample = torch.LongTensor([head, relation, tail])
        timestamp_tensor = torch.FloatTensor([timestamp])
        
        return positive_sample, negative_sample, filter_bias, self.mode, timestamp_tensor
    
    @staticmethod
    def collate_fn(data):
        positive_sample = torch.stack([_[0] for _ in data], dim=0)
        negative_sample = torch.stack([_[1] for _ in data], dim=0)
        filter_bias = torch.stack([_[2] for _ in data], dim=0)
        mode = data[0][3]
        timestamps = torch.cat([_[4] for _ in data], dim=0)
        
        return positive_sample, negative_sample, filter_bias, mode, timestamps


def load_temporal_data(data_path):
    """
    Load preprocessed temporal KG data
    
    Args:
        data_path: Path to processed data directory (e.g., 'processed/ICEWS14')
    
    Returns:
        Dictionary with:
        - train/valid/test: Lists of (h, r, t, ts_norm, ts_orig) tuples
        - entity2id/relation2id: ID mappings
        - nentity/nrelation: Entity/relation counts
        - metadata: Additional info
    """
    import os
    
    with open(os.path.join(data_path, 'entity2id.pkl'), 'rb') as f:
        entity2id = pickle.load(f)
    
    with open(os.path.join(data_path, 'relation2id.pkl'), 'rb') as f:
        relation2id = pickle.load(f)
    
    with open(os.path.join(data_path, 'train.pkl'), 'rb') as f:
        train_triples = pickle.load(f)
    
    with open(os.path.join(data_path, 'valid.pkl'), 'rb') as f:
        valid_triples = pickle.load(f)
    
    with open(os.path.join(data_path, 'test.pkl'), 'rb') as f:
        test_triples = pickle.load(f)
    
    with open(os.path.join(data_path, 'metadata.pkl'), 'rb') as f:
        metadata = pickle.load(f)
    
    nentity = len(entity2id)
    nrelation = len(relation2id)
    
    print(f'Loaded temporal KG from {data_path}')
    print(f'#Entities: {nentity}')
    print(f'#Relations: {nrelation}')
    print(f'#Train: {len(train_triples)}')
    print(f'#Valid: {len(valid_triples)}')
    print(f'#Test: {len(test_triples)}')
    
    return {
        'train': train_triples,
        'valid': valid_triples,
        'test': test_triples,
        'entity2id': entity2id,
        'relation2id': relation2id,
        'nentity': nentity,
        'nrelation': nrelation,
        'metadata': metadata
    }


def count_frequency(triples, start=4):
    """
    Get frequency of entities for subsampling weighting
    
    Args:
        triples: List of (h, r, t, ts, ts_orig) tuples
        start: Smoothing parameter (default 4)
    
    Returns:
        Dictionary mapping (entity, relation) to count
    """
    count = {}
    for sample in triples:
        if len(sample) == 4:
            head, relation, tail, timestamp = sample
        else:
            head, relation, tail, timestamp, _ = sample
        
        if (head, relation) not in count:
            count[(head, relation)] = start
        else:
            count[(head, relation)] += 1
        
        if (tail, -relation - 1) not in count:
            count[(tail, -relation - 1)] = start
        else:
            count[(tail, -relation - 1)] += 1
    
    return count
