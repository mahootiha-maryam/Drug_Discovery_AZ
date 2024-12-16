import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
import torch
import concurrent.futures
from torch.utils.data import Dataset, DataLoader
import numpy as np
from functools import lru_cache
import os

from rdkit import RDLogger
rd_logger = RDLogger.logger()
rd_logger.setLevel(RDLogger.CRITICAL)

neg_pairs_tr = pd.read_csv("neg_pairs_train_new.csv")
pos_pairs_tr = pd.read_csv("pos_pairs_train_new.csv")
neg_pairs_val = pd.read_csv("neg_pairs_val_new.csv")
pos_pairs_val = pd.read_csv("pos_pairs_val_new.csv")
neg_pairs_test = pd.read_csv("neg_pairs_test_new.csv")
pos_pairs_test = pd.read_csv("pos_pairs_test_new.csv")

# Add label column
neg_pairs_tr['label'] = 0
pos_pairs_tr['label'] = 1
neg_pairs_val['label'] = 0
pos_pairs_val['label'] = 1
neg_pairs_test['label'] = 0
pos_pairs_test['label'] = 1


# Concatenate the dataframes
train_df = pd.concat([neg_pairs_tr, pos_pairs_tr], axis=0, ignore_index=True)
val_df = pd.concat([neg_pairs_val, pos_pairs_val], axis=0, ignore_index=True)
test_df = pd.concat([neg_pairs_test, pos_pairs_test], axis=0, ignore_index=True)

def generate_fingerprint(smiles, radius=2, bits=1024):
    mol = Chem.MolFromSmiles(smiles)
    fingerprint = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=bits)
    return fingerprint

def fingerprint_rules(rule, fingerprint_size=2048):
    """
    Generates a dense fingerprint vector from a SMARTS reaction rule.

    Parameters:
    - rule (str): The SMARTS string defining the reaction.
    - fingerprint_size (int): The size of the resulting fingerprint vector.

    Returns:
    - np.ndarray: A dense fingerprint vector.
    """
    # Generate the reaction object from the SMARTS rule
    rxn = AllChem.ReactionFromSmarts(rule)
    if rxn is None:
        raise ValueError("Invalid SMARTS rule provided.")

    # Create the difference fingerprint for the reaction
    fingerprint_rule = Chem.rdChemReactions.CreateDifferenceFingerprintForReaction(rxn)

    # Initialize a NumPy array of zeros
    bit_array = np.zeros(fingerprint_size, dtype=np.float32)

    # Populate the bit array based on the non-zero entries
    for bit_id in fingerprint_rule.GetNonzeroElements().keys():
        if bit_id < fingerprint_size:
            bit_array[bit_id] = 1.0
        else:
            # Handle bits outside the fingerprint size if necessary
            # For example, you might log a warning or implement a hashing mechanism
            print(f"Warning: bit_id {bit_id} exceeds fingerprint size {fingerprint_size}. Ignored.")

    return bit_array
    
positive_train_df = train_df[train_df['label'] == 1]
negative_train_df = train_df[train_df['label'] == 0]
 
positive_val_df = val_df[val_df['label'] == 1]
negative_val_df = val_df[val_df['label'] == 0]

positive_test_df = test_df[test_df['label'] == 1]
negative_test_df = test_df[test_df['label'] == 0]


positive_train_df.reset_index(inplace=True, drop=True)
negative_train_df.reset_index(inplace=True, drop=True)

positive_val_df.reset_index(inplace=True, drop=True)
negative_val_df.reset_index(inplace=True, drop=True)

positive_test_df.reset_index(inplace=True, drop=True)
negative_test_df.reset_index(inplace=True, drop=True)


def generate_input_data_optimized(data_frame):
    inputs_b = []
    inputs_c = []
    inputs_r = []
    labels = []

    for i, row in data_frame.iterrows():
        mol_b = row['mol1']
        mol_c = row['mol2']
        rule = row['rule']

        fingerprint_b = generate_fingerprint(mol_b)
        fingerprint_c = generate_fingerprint(mol_c)

        feature_rule = fingerprint_rules(rule)

        inputs_b.append(fingerprint_b)
        inputs_c.append(fingerprint_c)
        inputs_r.append(feature_rule)
        labels.append(row['label']) 

    inputs_b = torch.tensor(inputs_b, dtype=torch.float32)
    inputs_c = torch.tensor(inputs_c, dtype=torch.float32)
    inputs_r = torch.tensor(inputs_r, dtype=torch.float32)
    labels = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)

    return inputs_b, inputs_c, inputs_r, labels

class ReactionDataset(torch.utils.data.Dataset):
    def __init__(self, inputs_b, inputs_c, inputs_r, labels):
        self.inputs_b = inputs_b
        self.inputs_c = inputs_c
        self.inputs_r = inputs_r
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.inputs_b[index], self.inputs_c[index], self.inputs_r[index], self.labels[index]

# 4. Caching Processed Datasets
def save_dataset(file_path, dataset):
    torch.save(dataset, file_path)

def load_or_generate_dataset(df, cache_path):

    print(f"Generating dataset and saving to {cache_path}")
    inputs_b, inputs_c, inputs_r, labels = generate_input_data_optimized(df)
    dataset = ReactionDataset(inputs_b, inputs_c, inputs_r, labels)
    save_dataset(cache_path, dataset)
    return dataset
    

positive_train_dataset = load_or_generate_dataset(
    positive_train_df, "positive_train_dataset_FPFP.pt"
)
print("Positive train dataset >> complete")

negative_train_dataset = load_or_generate_dataset(
    negative_train_df, "negative_train_dataset_FPFP.pt"
)
print("Negative train dataset >> complete")

positive_val_dataset = load_or_generate_dataset(
    positive_val_df, "positive_val_dataset_FPFP.pt"
)
print("Positive validation dataset >> complete")

negative_val_dataset = load_or_generate_dataset(
    negative_val_df, "negative_val_dataset_FPFP.pt"
)
print("Negative validation dataset >> complete")

positive_test_dataset = load_or_generate_dataset(
    positive_test_df, "positive_test_dataset_FPFP.pt"
)
print("Positive test dataset >> complete")

negative_test_dataset = load_or_generate_dataset(
    negative_test_df, "negative_test_dataset_FPFP.pt"
)

