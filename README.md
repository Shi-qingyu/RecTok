<div align="center">

# RecTok: Reconstruction Distillation along Rectified Flow

**Official PyTorch Implementation**

[![arXiv](https://img.shields.io/badge/arXiv-RecTok-b31b1b.svg?style=flat-square)]()
[![HuggingFace Model](https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-Model-yellow?style=flat-square)](https://huggingface.co/QingyuShi/RecTok)
[![Project Page](https://img.shields.io/badge/Project-Page-blue?style=flat-square)](https://shi-qingyu.github.io/rectok.github.io/)
[![License](https://img.shields.io/badge/License-Apache_2.0-green.svg?style=flat-square)](LICENSE)

</div>

---

## 🛠️ Preparation

### 1. Installation

Set up the environment and install dependencies:

```bash
# Clone the repository
git clone https://github.com/Shi-qingyu/RecTok.git
cd RecTok

# Create and activate conda environment
conda create -n rectok python=3.10 -y
conda activate rectok

# Install requirements
pip install -r requirements.txt
```

### 2. Download Models
Download pretrained models and necessary data assets:
```bash
# Download from HuggingFace
huggingface-cli download QingyuShi/RecTok --local-dir ./pretrained_models
# Organize data assets and offline models
mv ./pretrained_models/data ./data
mv ./pretrained_models/offline_models.zip ./offline_models.zip
unzip offline_models.zip && rm offline_models.zip
```

### 3. Download ImageNet-1K
Please download ImageNet-1K to `./data`. Your directory structure should look like this:
```text
data/
├── fid_stats/                          # FID statistics files
│   ├── adm_in256_stats.npz             # For gFID
│   ├── val_fid_statistics_file_256.npz # For rFID
│   └── val_fid_statistics_file_512.npz # For rFID
├── imagenet/                           # ImageNet dataset
│   ├── train/
│   │   ├── n01440764/
│   │   └── ...
│   └── val/
│       ├── n01440764/
│       └── ...
├── train.txt                           # Training file list
└── val.txt                             # Validation file list
```

---
## 📊 Evaluation
### Tokenizer Evaluation
Evaluate the reconstruction performance of the tokenizer:
```bash
bash run_eval_tokenizer.sh pretrained_models/RecTok_decft.pth
```
### Generative Model Evaluation
Evaluate the generation quality (FID, etc.):
```bash
bash run_eval_diffusion.sh \
    pretrained_models/RecTok_decft.pth \
    pretrained_models/ditdhxl_epoch_0599.pth \
    pretrained_models/ditdhs_epoch_0029.pth
```
---

## 🚀 Training

### 1. Tokenizer Training
**Stage 1: Train RecTok Tokenizer**
```bash
bash run_train_tokenizer.sh
```
**Stage 2: Decoder Fine-tuning**
Run the following command for decoder fine-tuning:
```bash
bash run_decoder_finetune_tokenizer.sh <exp_name in Stage 1 run_train_tokenizer.sh>
```
### 2. Generative Model Training
**Option A: Train from Scratch:**
Train the diffusion transformer model ($\text{DiT}^{\text{DH}}\text{-XL}$):
```bash
bash run_train_diffusion.sh <exp_name in Stage 2 run_decoder_finetune_tokenizer.sh>
```

**Option B: Train with Pretrained RecTok:**
To train DiT based on our official pretrained RecTok weights:
```bash
mkdir -p work_dirs/tokenizer_training/rectok/checkpoints
cp pretrained_models/RecTok_decft.pth work_dirs/tokenizer_training/rectok/checkpoints/latest.pth
bash run_train_diffusion.sh rectok
```

---
## 📜 Citation
If you find this work useful for your research, please consider citing:
```bibtex
placeholder
```

---
## 🙏 Acknowledgements
We thank the authors of [lDeTok](https://github.com/Jiawei-Yang/DeTok), [RAE](https://github.com/bytetriper/RAE), [MAE](https://github.com/facebookresearch/mae), [DiT](https://github.com/facebookresearch/DiT), and [LightningDiT](https://github.com/hustvl/LightningDiT) for their foundational work.

Our codebase builds upon several excellent open-source projects, including [lDeTok](https://github.com/Jiawei-Yang/DeTok), [RAE](https://github.com/bytetriper/RAE), and [torch_fidelity](https://github.com/toshas/torch-fidelity). We are grateful to the communities behind them.