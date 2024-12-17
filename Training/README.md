# Training Models for Contrastive Learning

In this directory, we explore different methodologies for training a **contrastive learning model**. Here's how we approached feature extraction and model training:

## Feature Extraction

### Molecule Features
- **Fingerprints**: Traditional method for encoding molecular structures into a vector format.
- **Pretrained Transformer**: Utilizes the last layer of a transformer model, pre-trained on a vast molecular dataset, to extract rich molecular embeddings.

### Reaction Rule Features
- **One-Hot Vector**: A binary vector representation where each bit indicates the presence or absence of a specific reaction rule.
- **Difference Fingerprint**: Captures the chemical changes between reactants and products, providing a more nuanced view of reaction dynamics.

## Model Architectures

We have trained three distinct models, each with a different approach to feature representation:

### Model1
- **Molecule Features**: Fingerprints
- **Reaction Rule Features**: One-Hot Vectors

### Model2
- **Molecule Features**: Fingerprints
- **Reaction Rule Features**: Difference Fingerprints

### Model3
- **Molecule Features**: Pretrained Transformer Encoder Output
- **Reaction Rule Features**: One-Hot Vectors
