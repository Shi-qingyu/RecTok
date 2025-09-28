#!/bin/bash

# Script to unzip all .zip files in ./data/imagenet/train/ directory
# Author: Auto-generated script
# Usage: ./unzip_imagenet_train.sh

set -e  # Exit on any error

mkdir data/
huggingface-cli download QingyuShi/ImageNet1K --local-dir ./data/imagenet/ --repo-type dataset

cd data/imagenet/
unzip val.zip

cd train

# Collect zips (non-recursive)
zips=( *.zip )

if ((${#zips[@]} == 0)); then
  echo "[info] No .zip files found in: $(pwd)"
  exit 0
fi

# Extract each zip into the current directory (no -d specified)
for z in "${zips[@]}"; do
  echo "[proc] unzip '$z' -> $(pwd)"
  # Use -o to overwrite without prompting. Remove -o if you prefer interactive prompts, or use -n to never overwrite.
  unzip -o -- "$z"
done

echo "[done] All zips processed."

cd ../../../

# Download everything from huggingface
huggingface-cli download jjiaweiyang/l-DeTok --local-dir released_model
mv released_model/train.txt data/
mv released_model/val.txt data/
mv released_model/fid_stats data/
mv released_model/imagenet-val-prc.zip ./data/

# Unzip: imagenet-val-prc.zip for precision & recall evaluation
python -m zipfile -e ./data/imagenet-val-prc.zip ./data/