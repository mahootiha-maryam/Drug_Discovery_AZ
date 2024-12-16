# Dataset Preparation for Contrastive Learning Model

This repository contains the necessary scripts for preparing datasets to train a contrastive learning model for chemical reactions.

## 📁 Repository Contents

### 1. Making_pairs.ipynb
Jupyter notebook for generating positive and negative molecular pairs for training. For a detailed explanation of the pair generation process, see [Pair Generation Documentation](https://github.com/mahootiha-maryam/Drug_Discovery_AZ/blob/main/Making_Dataset/Making_Pairs.ipynb).

### 2. Dataset Generation Scripts

#### 🧬 making_dataset_FP_OhE.py
Creates datasets using Morgan fingerprints and one-hot encoded reaction rules.

**Features:**
- Molecular representation: Morgan fingerprints (size: 1024)
- Reaction rules: One-hot encoding (size: 56)

#### 🔄 making_dataset_FP_diff.py
Generates datasets using Morgan fingerprints and fingerprint differences for reactions.

**Features:**
- Molecular representation: Morgan fingerprints (size: 1024)
- Reaction representation: Fingerprint differences (size: 2048)
