#!/bin/bash
# Download ICEWS14 dataset for Baseline PairRE
# Data from Facebook AI Research (TKBC)

echo "=================================================="
echo "Downloading ICEWS14 Dataset"
echo "=================================================="

# Get script directory
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"

URL="https://dl.fbaipublicfiles.com/tkbc/data.tar.gz"
OUTPUT="data.tar.gz"

echo "Downloading from $URL ..."
wget -q --show-progress "$URL" -O "$OUTPUT"

if [ $? -ne 0 ]; then
    echo "wget failed, trying curl..."
    curl -L --progress-bar "$URL" -o "$OUTPUT"
fi

echo "Extracting data.tar.gz ..."
tar -xzf "$OUTPUT"

echo "Checking archive contents..."
echo "Available datasets:"
ls -la src_data/

# TKBC dataset structure: src_data/ICEWS14/
if [ -d "src_data/ICEWS14" ]; then
    echo "Found: src_data/ICEWS14"
    echo ""
    echo "Contents of src_data/ICEWS14/:"
    ls -lh src_data/ICEWS14/
    echo ""
    
    # TKBC might have different file structure
    # Check if .txt files exist or if there are .pkl files already
    mkdir -p raw
    
    if [ -f "src_data/ICEWS14/train.txt" ]; then
        echo "Found .txt files - copying to raw/..."
        cp src_data/ICEWS14/*.txt raw/
        need_preprocess=true
    elif [ -f "src_data/ICEWS14/train" ]; then
        echo "Found files without extension - copying to raw/ with .txt extension..."
        cp src_data/ICEWS14/train raw/train.txt
        cp src_data/ICEWS14/valid raw/valid.txt
        cp src_data/ICEWS14/test raw/test.txt
        echo "✓ Copied train, valid, test → raw/*.txt"
        need_preprocess=true
    elif [ -f "src_data/ICEWS14/train.pickle" ] || [ -f "src_data/ICEWS14/train.pkl" ]; then
        echo "Found .pkl/.pickle files - data already processed!"
        mv src_data/ICEWS14 processed
        need_preprocess=false
    else
        echo "✗ Unknown data format in ICEWS14 folder"
        echo "Contents:"
        ls -la src_data/ICEWS14/
        exit 1
    fi
    
    rm -rf src_data
    rm "$OUTPUT"
    
    if [ "$need_preprocess" = true ]; then
        echo ""
        echo "Preprocessing data to create .pkl files..."
        python preprocess_data.py --raw_dir raw --output_dir processed
        
        if [ $? -eq 0 ]; then
            echo ""
            echo "✓ Preprocessing complete!"
        else
            echo "✗ Preprocessing failed!"
            exit 1
        fi
    fi
    
    echo ""
    echo "✓ Data ready in ./processed/"
    echo ""
    echo "Contents:"
    ls -lh processed/
else
    echo "✗ ICEWS14 not found in archive"
    echo "Available folders:"
    ls -la src_data/ 2>/dev/null || ls -la
    exit 1
fi

echo ""
echo "=================================================="
echo "Dataset Ready!"
echo "=================================================="
echo "You can now run training:"
echo "  bash train_baseline.sh"
echo "  OR"
echo "  python run.py --do_train --cuda --data_path processed ..."
echo "=================================================="
