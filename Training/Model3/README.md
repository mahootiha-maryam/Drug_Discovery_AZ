# Contrastive Model 3 Training

This directory focuses on training a **contrastive model** leveraging molecule features extracted from a pretrained transformer and rule features represented by one-hot vectors.

## Overview

### Data Preparation
- **Pair Selection**: Before training, we filtered the existing pairs to ensure all SMILES tokens are within the vocabulary of our pretrained transformer, thus avoiding errors during feature extraction through [this file](https://github.com/mahootiha-maryam/Drug_Discovery_AZ/blob/main/Training/Model3/Make_goodpairs.ipynb)

### Model Training
- **Main Training Script**: `Train_lightning.py`
  - Instead of pre-computing and storing a large dataset, features are extracted **during** the training loop to conserve computational resources and storage.

### Feature Extraction
- **Molecule Features**:
  - Extracted using a **pretrained transformer** from [MolecularAI/exahustive_search_mol2mol](https://github.com/MolecularAI/exahustive_search_mol2mol/tree/main/paper_checkpoints). The transformer is frozen, and only the last encoder layer's output is used for features.
  - **Mol_Feature.py**: This script handles molecule feature extraction:
    - **Zero out padding** to ensure only meaningful tokens contribute to the feature vector.
    - **Mean pooling** across tokens to accommodate varying token counts per molecule.
    - **Setup**: Ensure a directory named `MolAI` with a subdirectory `main` contains the dataset and model directories from [exahustive_search_mol2mol/lib](https://github.com/MolecularAI/exahustive_search_mol2mol/tree/main/lib).

- **Rule Features**:
  - Extracted using **one-hot vectors** via `Rule_Feature.py`.

### Model Architecture
- **Network.py**: Defines the architecture for the contrastive learning model.
