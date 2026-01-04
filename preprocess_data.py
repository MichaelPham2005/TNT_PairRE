#!/usr/bin/env python3
"""
Preprocess ICEWS14 raw data (.txt) to processed format (.pkl)
Converts from TKBC format to PairRE format
"""

import os
import pickle
import argparse
from collections import defaultdict
from datetime import datetime

def read_triplets(file_path):
    """Read triplets from text file."""
    triplets = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t')
            if len(parts) == 4:
                h, r, t, timestamp = parts
                triplets.append((h, r, t, timestamp))
            else:
                print(f"Warning: Skipping malformed line: {line}")
    return triplets

def build_mappings(train_data, valid_data, test_data):
    """Build entity and relation ID mappings."""
    entities = set()
    relations = set()
    
    for h, r, t, _ in train_data + valid_data + test_data:
        entities.add(h)
        entities.add(t)
        relations.add(r)
    
    # Sort for consistency
    entities = sorted(list(entities))
    relations = sorted(list(relations))
    
    entity2id = {e: i for i, e in enumerate(entities)}
    relation2id = {r: i for i, r in enumerate(relations)}
    
    return entity2id, relation2id

def normalize_timestamps(timestamps):
    """Normalize timestamps to [0, 1] range."""
    # Convert date strings to Unix timestamps
    numeric_timestamps = []
    for ts in timestamps:
        if isinstance(ts, str):
            # Parse date string: '2014-05-13' or '2014-05-13 12:00:00'
            try:
                dt = datetime.strptime(ts.split()[0], '%Y-%m-%d')
                numeric_timestamps.append(dt.timestamp())
            except Exception as e:
                print(f"Warning: Could not parse timestamp '{ts}': {e}")
                numeric_timestamps.append(0.0)
        else:
            numeric_timestamps.append(float(ts))
    
    min_ts = min(numeric_timestamps)
    max_ts = max(numeric_timestamps)
    
    if max_ts == min_ts:
        return [0.0] * len(numeric_timestamps), min_ts, max_ts
    
    normalized = [(t - min_ts) / (max_ts - min_ts) for t in numeric_timestamps]
    return normalized, min_ts, max_ts

def convert_to_ids(triplets, entity2id, relation2id, timestamps_norm):
    """Convert triplets to ID format with normalized timestamps."""
    converted = []
    for i, (h, r, t, _) in enumerate(triplets):
        h_id = entity2id[h]
        r_id = relation2id[r]
        t_id = entity2id[t]
        ts_norm = timestamps_norm[i]
        converted.append((h_id, r_id, t_id, ts_norm))
    return converted

def build_temporal_filter(train_data, valid_data, test_data):
    """
    Build to_skip.pkl for temporal filtering.
    Format: {(h, r, t): set(timestamps)} - all valid (h,r,t,t) combinations
    """
    all_facts = defaultdict(set)
    
    # Collect all (h,r,t,timestamp) from train/valid/test
    for h, r, t, ts in train_data + valid_data + test_data:
        all_facts[(h, r, t)].add(ts)
    
    return dict(all_facts)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_dir', type=str, default='raw',
                       help='Directory with raw .txt files')
    parser.add_argument('--output_dir', type=str, default='processed',
                       help='Output directory for .pkl files')
    args = parser.parse_args()
    
    print("="*60)
    print("ICEWS14 Data Preprocessing")
    print("="*60)
    
    # Read raw data
    print("\n1. Reading raw data...")
    train_file = os.path.join(args.raw_dir, 'train.txt')
    valid_file = os.path.join(args.raw_dir, 'valid.txt')
    test_file = os.path.join(args.raw_dir, 'test.txt')
    
    if not os.path.exists(train_file):
        print(f"Error: {train_file} not found!")
        return
    
    train_raw = read_triplets(train_file)
    valid_raw = read_triplets(valid_file)
    test_raw = read_triplets(test_file)
    
    print(f"  Train: {len(train_raw)} triplets")
    print(f"  Valid: {len(valid_raw)} triplets")
    print(f"  Test:  {len(test_raw)} triplets")
    
    # Build mappings
    print("\n2. Building entity/relation mappings...")
    entity2id, relation2id = build_mappings(train_raw, valid_raw, test_raw)
    print(f"  Entities: {len(entity2id)}")
    print(f"  Relations: {len(relation2id)}")
    
    # Normalize timestamps
    print("\n3. Normalizing timestamps...")
    all_timestamps = [ts for _, _, _, ts in train_raw + valid_raw + test_raw]
    norm_timestamps, min_ts, max_ts = normalize_timestamps(all_timestamps)
    
    print(f"  Original range: [{min_ts}, {max_ts}]")
    print(f"  Normalized to: [0.0, 1.0]")
    
    # Split normalized timestamps
    train_ts = norm_timestamps[:len(train_raw)]
    valid_ts = norm_timestamps[len(train_raw):len(train_raw)+len(valid_raw)]
    test_ts = norm_timestamps[len(train_raw)+len(valid_raw):]
    
    # Convert to IDs
    print("\n4. Converting to ID format...")
    train_data = convert_to_ids(train_raw, entity2id, relation2id, train_ts)
    valid_data = convert_to_ids(valid_raw, entity2id, relation2id, valid_ts)
    test_data = convert_to_ids(test_raw, entity2id, relation2id, test_ts)
    
    # Build temporal filter
    print("\n5. Building temporal filter (to_skip.pkl)...")
    to_skip = build_temporal_filter(train_data, valid_data, test_data)
    print(f"  Unique (h,r,t) facts: {len(to_skip)}")
    
    # Save processed data
    print("\n6. Saving processed data...")
    os.makedirs(args.output_dir, exist_ok=True)
    
    with open(os.path.join(args.output_dir, 'train.pkl'), 'wb') as f:
        pickle.dump(train_data, f)
    
    with open(os.path.join(args.output_dir, 'valid.pkl'), 'wb') as f:
        pickle.dump(valid_data, f)
    
    with open(os.path.join(args.output_dir, 'test.pkl'), 'wb') as f:
        pickle.dump(test_data, f)
    
    # Save mappings and metadata
    mappings = {
        'nentity': len(entity2id),
        'nrelation': len(relation2id),
        'entity2id': entity2id,
        'relation2id': relation2id,
        'id2entity': {v: k for k, v in entity2id.items()},
        'id2relation': {v: k for k, v in relation2id.items()},
        'min_timestamp': min_ts,
        'max_timestamp': max_ts
    }
    
    with open(os.path.join(args.output_dir, 'mappings.pkl'), 'wb') as f:
        pickle.dump(mappings, f)
    
    with open(os.path.join(args.output_dir, 'to_skip.pkl'), 'wb') as f:
        pickle.dump(to_skip, f)
    
    print(f"\n✓ Processed data saved to: {args.output_dir}/")
    print(f"  - train.pkl ({len(train_data)} samples)")
    print(f"  - valid.pkl ({len(valid_data)} samples)")
    print(f"  - test.pkl ({len(test_data)} samples)")
    print(f"  - mappings.pkl (entity/relation IDs)")
    print(f"  - to_skip.pkl (temporal filtering)")
    
    print("\n" + "="*60)
    print("Preprocessing Complete!")
    print("="*60)
    print("You can now run training:")
    print("  python run.py --do_train --cuda --data_path processed ...")
    print("="*60)

if __name__ == '__main__':
    main()
