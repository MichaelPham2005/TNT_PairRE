#!/usr/bin/python3

"""
Download and prepare ICEWS14 dataset for Baseline PairRE
"""

import os
import urllib.request
import tarfile
import pickle
import numpy as np
from collections import defaultdict


def download_icews14(data_dir='data'):
    """
    Check for ICEWS14 dataset files
    User should manually place train.txt, valid.txt, test.txt in data_dir
    """
    os.makedirs(data_dir, exist_ok=True)
    
    required_files = ['train.txt', 'valid.txt', 'test.txt']
    
    print("Checking for ICEWS14 dataset files...")
    missing_files = []
    
    for filename in required_files:
        file_path = os.path.join(data_dir, filename)
        if os.path.exists(file_path):
            print(f"  ✓ Found {filename}")
        else:
            missing_files.append(filename)
            print(f"  ✗ Missing {filename}")
    
    if missing_files:
        print("\n" + "="*60)
        print("⚠️  Missing ICEWS14 data files!")
        print("="*60)
        print("\nPlease download ICEWS14 dataset and place these files in 'data/' folder:")
        print("  - train.txt")
        print("  - valid.txt")
        print("  - test.txt")
        print("\nData sources:")
        print("  1. https://github.com/INK-USC/RE-Net/tree/master/data/ICEWS14")
        print("  2. https://github.com/woojeongjin/DE-SimplE")
        print("  3. Or from temporal/ folder if you have it:")
        print("     cp ../temporal/src_data/ICEWS14/raw/*.txt data/")
        print("\nFormat: Each line is 'head\\trelation\\ttail\\ttimestamp'")
        print("="*60)
        return False
    
    print("\n✓ All required files found!")
    return True


def load_raw_data(data_dir='data'):
    """
    Load raw ICEWS14 data from txt files
    Format: head    relation    tail    timestamp
    """
    print("\nLoading raw data...")
    
    def load_file(filename):
        triples = []
        with open(os.path.join(data_dir, filename), 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 4:
                    head, rel, tail, timestamp = parts
                    triples.append((head, rel, tail, timestamp))
        return triples
    
    train = load_file('train.txt')
    valid = load_file('valid.txt')
    test = load_file('test.txt')
    
    print(f"  Train: {len(train)} quadruples")
    print(f"  Valid: {len(valid)} quadruples")
    print(f"  Test: {len(test)} quadruples")
    
    return train, valid, test


def build_mappings(train, valid, test):
    """
    Build entity and relation mappings to IDs
    """
    print("\nBuilding entity and relation mappings...")
    
    entities = set()
    relations = set()
    timestamps = set()
    
    for dataset in [train, valid, test]:
        for head, rel, tail, timestamp in dataset:
            entities.add(head)
            entities.add(tail)
            relations.add(rel)
            timestamps.add(timestamp)
    
    # Sort for deterministic ordering
    entities = sorted(entities)
    relations = sorted(relations)
    timestamps = sorted(timestamps)
    
    entity2id = {e: i for i, e in enumerate(entities)}
    relation2id = {r: i for i, r in enumerate(relations)}
    
    # Normalize timestamps to [0, 1]
    timestamps = sorted([int(t) for t in timestamps])
    min_time = min(timestamps)
    max_time = max(timestamps)
    time_range = max_time - min_time if max_time > min_time else 1
    
    timestamp2norm = {str(t): (t - min_time) / time_range for t in timestamps}
    
    print(f"  Entities: {len(entities)}")
    print(f"  Relations: {len(relations)}")
    print(f"  Timestamps: {len(timestamps)}")
    print(f"  Time range: {min_time} - {max_time}")
    
    return entity2id, relation2id, timestamp2norm


def process_triples(raw_triples, entity2id, relation2id, timestamp2norm):
    """
    Convert raw triples to ID format with normalized timestamps
    Returns: List of (head_id, relation_id, tail_id, normalized_time, original_time)
    """
    processed = []
    for head, rel, tail, timestamp in raw_triples:
        head_id = entity2id[head]
        rel_id = relation2id[rel]
        tail_id = entity2id[tail]
        norm_time = timestamp2norm[timestamp]
        orig_time = int(timestamp)
        
        processed.append((head_id, rel_id, tail_id, norm_time, orig_time))
    
    return processed


def save_processed_data(train, valid, test, entity2id, relation2id, save_dir='processed'):
    """
    Save processed data in pickle format
    """
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"\nSaving processed data to {save_dir}/...")
    
    # Save triples
    with open(os.path.join(save_dir, 'train.pkl'), 'wb') as f:
        pickle.dump(train, f)
    print(f"  ✓ Saved train.pkl ({len(train)} quadruples)")
    
    with open(os.path.join(save_dir, 'valid.pkl'), 'wb') as f:
        pickle.dump(valid, f)
    print(f"  ✓ Saved valid.pkl ({len(valid)} quadruples)")
    
    with open(os.path.join(save_dir, 'test.pkl'), 'wb') as f:
        pickle.dump(test, f)
    print(f"  ✓ Saved test.pkl ({len(test)} quadruples)")
    
    # Save mappings
    mappings = {
        'entity2id': entity2id,
        'relation2id': relation2id,
        'nentity': len(entity2id),
        'nrelation': len(relation2id)
    }
    
    with open(os.path.join(save_dir, 'mappings.pkl'), 'wb') as f:
        pickle.dump(mappings, f)
    print(f"  ✓ Saved mappings.pkl")
    
    # Save statistics
    stats = {
        'nentity': len(entity2id),
        'nrelation': len(relation2id),
        'ntrain': len(train),
        'nvalid': len(valid),
        'ntest': len(test)
    }
    
    with open(os.path.join(save_dir, 'stats.txt'), 'w') as f:
        for key, value in stats.items():
            f.write(f"{key}: {value}\n")
    print(f"  ✓ Saved stats.txt")
    
    print("\n✓ Data preparation complete!")


def prepare_icews14(data_dir='data', processed_dir='processed'):
    """
    Main function to download and prepare ICEWS14 dataset
    """
    print("=" * 60)
    print("ICEWS14 Data Preparation for Baseline PairRE")
    print("=" * 60)
    
    # Step 1: Download
    if not download_icews14(data_dir):
        print("\n✗ Download failed. Please check and try again.")
        return False
    
    # Step 2: Load raw data
    train_raw, valid_raw, test_raw = load_raw_data(data_dir)
    
    # Step 3: Build mappings
    entity2id, relation2id, timestamp2norm = build_mappings(train_raw, valid_raw, test_raw)
    
    # Step 4: Process triples
    print("\nProcessing triples...")
    train_processed = process_triples(train_raw, entity2id, relation2id, timestamp2norm)
    valid_processed = process_triples(valid_raw, entity2id, relation2id, timestamp2norm)
    test_processed = process_triples(test_raw, entity2id, relation2id, timestamp2norm)
    
    # Step 5: Save
    save_processed_data(train_processed, valid_processed, test_processed,
                       entity2id, relation2id, processed_dir)
    
    print("\n" + "=" * 60)
    print("Dataset Statistics:")
    print("=" * 60)
    print(f"Entities: {len(entity2id)}")
    print(f"Relations: {len(relation2id)}")
    print(f"Train quadruples: {len(train_processed)}")
    print(f"Valid quadruples: {len(valid_processed)}")
    print(f"Test quadruples: {len(test_processed)}")
    print("=" * 60)
    
    return True


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Download and prepare ICEWS14 dataset')
    parser.add_argument('--data_dir', type=str, default='data',
                       help='Directory to store raw data')
    parser.add_argument('--processed_dir', type=str, default='processed',
                       help='Directory to store processed data')
    
    args = parser.parse_args()
    
    success = prepare_icews14(args.data_dir, args.processed_dir)
    
    if success:
        print("\n✓ Success! You can now run training with:")
        print(f"  python run.py --data_path {args.processed_dir} --do_train --cuda ...")
    else:
        print("\n✗ Failed to prepare dataset.")
