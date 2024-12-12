# Enhancing Molecular Synthesizability in ReInvent

## Overview

This repository focuses on integrating a **synthesizability policy** into the **ReInvent** framework for molecular design. The primary goal is to address a key limitation of ReInvent, where it might generate molecules that are theoretically interesting but practically challenging or infeasible to synthesize in a lab setting.

## Project Description

### Problem Statement
- **ReInvent Limitation**: While ReInvent excels at proposing novel molecular structures, it often overlooks the practical aspects of synthesis, such as:
  - Availability of reagents
  - Reaction conditions
  - Feasibility of chemical transformations

### Proposed Solution
- **Synthesizability Policy**: We introduce a policy to guide the reinforcement learning model of ReInvent to consider synthesis feasibility:
  - **Probability Model**: The probability of generating molecule A is modeled as:
    ```plaintext
    P(A) = P(B, C, R1) = P(B|C, R1) * P(R1|C) * P(C)
    ```
  - **Model Input**: Triplets of `(molecule B, molecule C, reaction R1)`.

### Methodology

#### Training Model
- **Objective**: Train a model to predict the probability of observing molecule B given fixed molecule C and reaction R1.
- **Approach**: Use **deep learning** with **contrastive learning** to differentiate between:
  - **Positive Pairs**: Molecules B that can react with C under R1.
  - **Negative Pairs**: Molecules B that do not react as expected.

#### Model Details
- **Loss Function**: Triplet loss with a margin to maximize the distance between negative pairs and minimize for positive pairs.
- **Output**: A value between 0 and 1:
  - Closer to **0**: Indicates a negative pair (unlikely to react).
  - Closer to **1**: Indicates a positive pair (likely to react).

## Repository Structure

```plaintext
├── data/            # Datasets for training and validation
├── models/          # Trained models
├── notebooks/       # Jupyter notebooks with experiments and analysis
├── src/             # Source code for model training and evaluation
│   ├── data_loader.py
│   ├── model.py
│   └── train.py
└── README.md
