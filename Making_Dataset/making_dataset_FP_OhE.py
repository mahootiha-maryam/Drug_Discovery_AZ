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


unique_rules = np.unique(pd.concat([train_df['rule'],val_df['rule'], test_df['rule']]))
num_rules = len(unique_rules)

rule_to_number = {rule: idx+1 for idx, rule in enumerate(unique_rules)}

train_df['rule_number'] = train_df['rule'].apply(lambda x: rule_to_number[x])
val_df['rule_number'] = val_df['rule'].apply(lambda x: rule_to_number[x])
test_df['rule_number'] = test_df['rule'].apply(lambda x: rule_to_number[x])

def generate_rule_feature_vector(rule, total_rules):
    # Initialize a zero vector
    vector = [0] * total_rules

    if rule is not None:
        if 1 <= rule <= total_rules:
            # Set the corresponding rule index to 1
            vector[rule - 1] = 1
        else:
            raise ValueError(f"Rule number {rule} is out of the valid range 1-{total_rules}.")
    
    return vector

features_reaction_train = []
features_reaction_val = []
features_reaction_test = []

total_rules = len(rule_to_number)

for rule in train_df['rule_number']:
    features_reaction_train.append(generate_rule_feature_vector(rule, total_rules))
    
for rule in val_df['rule_number']:
    features_reaction_val.append(generate_rule_feature_vector(rule, total_rules))
    
for rule in test_df['rule_number']:
    features_reaction_test.append(generate_rule_feature_vector(rule, total_rules))

positive_train_df = train_df[train_df['label'] == 1]
negative_train_df = train_df[train_df['label'] == 0]
 
positive_val_df = val_df[val_df['label'] == 1]
negative_val_df = val_df[val_df['label'] == 0]

positive_test_df = test_df[test_df['label'] == 1]
negative_test_df = test_df[test_df['label'] == 0]

features_reaction_train_pos = [features_reaction_train[i] for i in positive_train_df.index]
features_reaction_train_neg = [features_reaction_train[i] for i in negative_train_df.index]

features_reaction_val_pos = [features_reaction_val[i] for i in positive_val_df.index]
features_reaction_val_neg = [features_reaction_val[i] for i in negative_val_df.index]

features_reaction_test_pos = [features_reaction_test[i] for i in positive_test_df.index]
features_reaction_test_neg = [features_reaction_test[i] for i in negative_test_df.index]

positive_train_df.reset_index(inplace=True, drop=True)
negative_train_df.reset_index(inplace=True, drop=True)

positive_val_df.reset_index(inplace=True, drop=True)
negative_val_df.reset_index(inplace=True, drop=True)

positive_test_df.reset_index(inplace=True, drop=True)
negative_test_df.reset_index(inplace=True, drop=True)


# 1. Optimize Fingerprint Generation with Caching
@lru_cache(maxsize=None)
def generate_fingerprint_cached(mol):
    return generate_fingerprint(mol)

def generate_fingerprints_parallel(molecules, max_workers=8):
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        fingerprints = list(executor.map(generate_fingerprint_cached, molecules))
    return fingerprints

# 2. Vectorized Input Data Generation
def generate_input_data_optimized(data_frame, feature_vector):
    inputs_b = []
    inputs_c = []
    inputs_r = []
    labels = []

    for i, row in data_frame.iterrows():
        mol_b = row['mol1']
        mol_c = row['mol2']

        fingerprint_b = generate_fingerprint(mol_b)
        fingerprint_c = generate_fingerprint(mol_c)

        feature_rule = feature_vector[i]

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

def load_or_generate_dataset(df, feature_vector, cache_path, max_workers=8):
    if os.path.exists(cache_path):
        print(f"Loading cached dataset from {cache_path}")
        return torch.load(cache_path)
    else:
        print(f"Generating dataset and saving to {cache_path}")
        inputs_b, inputs_c, inputs_r, labels = generate_input_data_optimized(df, feature_vector)
        dataset = ReactionDataset(inputs_b, inputs_c, inputs_r, labels)
        save_dataset(cache_path, dataset)
        return dataset
    

positive_train_dataset = load_or_generate_dataset(
    positive_train_df, features_reaction_train_pos, "positive_train_dataset_new.pt"
)
print("Positive train dataset >> complete")

negative_train_dataset = load_or_generate_dataset(
    negative_train_df, features_reaction_train_neg, "negative_train_dataset_new.pt"
)
print("Negative train dataset >> complete")

positive_val_dataset = load_or_generate_dataset(
    positive_val_df, features_reaction_val_pos, "positive_val_dataset_new.pt"
)
print("Positive validation dataset >> complete")

negative_val_dataset = load_or_generate_dataset(
    negative_val_df, features_reaction_val_neg, "negative_val_dataset_new.pt"
)
print("Negative validation dataset >> complete")

positive_test_dataset = load_or_generate_dataset(
    positive_test_df, features_reaction_test_pos, "positive_test_dataset_new.pt"
)
print("Positive test dataset >> complete")

negative_test_dataset = load_or_generate_dataset(
    negative_test_df, features_reaction_test_neg, "negative_test_dataset_new.pt"
)
