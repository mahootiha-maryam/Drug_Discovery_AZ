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
import sys
from time import time
from functools import partial
import yaml
import sys
import os
from tqdm import tqdm
import torch.nn as nn
sys.path.append('/projects/mai/se_mai/users/kvvq085_Mary')
from MolAI.main.lib.model.search import beamsearch, LogicalOr, MaxLength, EOS, Node
from MolAI.main.lib.dataset.chem import standardize_smiles, remove_isotopes
from MolAI.main.lib.model.model import LitMolformer
from MolAI.main.lib.dataset.pair_dataset import PairedDataset
from rdkit import RDLogger


rd_logger = RDLogger.logger()
rd_logger.setLevel(RDLogger.CRITICAL)

sys.path.append('/projects/mai/se_mai/users/kvvq085_Mary/MolAI/main')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(config_path, checkpoint_path, vocabulary_path, device="cuda"):
    hparams = yaml.load(open(config_path), Loader=yaml.FullLoader)
    hparams["vocabulary"] = vocabulary_path
    model = LitMolformer(**hparams)
    state_dict = torch.load(checkpoint_path, map_location=device)["state_dict"]
    model.load_state_dict(state_dict)
    # Move the model to the device after loading the state dict
    model = model.to(device)
    model = model.eval()
    if "with_counts" in config_path:
        model.mol_to_fingerprints = partial(AllChem.GetMorganFingerprint, radius=2)
    else:
        model.mol_to_fingerprints = partial(
            AllChem.GetMorganFingerprintAsBitVect, radius=2, nBits=1024
        )
    return model

def generate_samples(model, smiles, beam_size=1000, device="cuda"):
    mol = Chem.MolFromSmiles(smiles)
    smi_no_iso = remove_isotopes(mol)
    smiles = standardize_smiles(smi_no_iso)
    src = model.vocabulary.encode(model.tokenizer.tokenize(smiles))
    src = torch.from_numpy(src.astype(np.int64)).to(device)
    src, src_mask, _, _, _ = PairedDataset.collate_fn([(src, src, torch.ones((1, 1)).to(device))])

    # Move src and src_mask to device after collate_fn
    src = src.to(device)
    src_mask = src_mask.to(device)

    samples, enc_out = sample_x(
        model,
        src[:1],
        src_mask[:1],
        decode_type="multinomial",
        beam_size=beam_size,
        beam_search_bs=512,
        device=device,  # Use the variable device here
    )

    return enc_out.cpu()

def subsequent_mask(size, device):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype("uint8")
    return torch.from_numpy(subsequent_mask).to(device) == 0

@torch.no_grad()
def sample_x(
    model,
    src,
    src_mask,
    decode_type,
    beam_size=None,
    beam_search_bs=None,
    device=device,
):
    vocabulary = model.vocabulary
    tokenizer = model.tokenizer
    result = None
    encoder_outputs = None
    if decode_type == "beamsearch":
        stop_criterion = LogicalOr((MaxLength(model.max_sequence_length - 1), EOS()))
        node = Node(
            model.network,
            (src, src_mask),
            vocabulary,
            device,
            batch_size=beam_search_bs,
            data_device=device,
        )
        beamsearch(node, beam_size, stop_criterion)
        output_smiles_list = [
            tokenizer.untokenize(vocabulary.decode(seq)) for seq in node.y.detach().cpu().numpy()
        ]
        input_smiles_list = []
        for seq in src.detach().cpu().numpy():
            s = tokenizer.untokenize(model.vocabulary.decode(seq))
            for _ in range(beam_size):
                input_smiles_list.append(s)
        nlls = (-node.loglikelihood.detach().cpu().numpy()).ravel()
        result = (input_smiles_list, output_smiles_list, nlls.tolist())
    else:
        batch_size = src.shape[0]
        ys = model.vocabulary.bos_token * torch.ones(1).to(device)
        ys = ys.repeat(batch_size, 1).view(batch_size, 1).type_as(src.data)
        encoder_outputs = model.network.encode(src, src_mask)
        break_condition = torch.zeros(batch_size, dtype=torch.bool).to(device)
        nlls = torch.zeros(batch_size).to(device)
        end_token = vocabulary.eos_token
        for i in range(model.max_sequence_length - 1):
            out = model.network.decode(
                encoder_outputs,
                src_mask,
                ys,
                subsequent_mask(ys.size(1), device).type_as(src.data),
            )
            log_prob = model.network.generator(out[:, -1])
            prob = torch.exp(log_prob)

            if decode_type == "greedy":
                _, next_word = torch.max(prob, dim=1)
                next_word = next_word.masked_fill(break_condition.to(device), 0)
                ys = torch.cat([ys, next_word.unsqueeze(-1)], dim=1)
                nlls += model._nll_loss(log_prob, next_word)
            elif decode_type == "multinomial":
                next_word = torch.multinomial(prob, 1)
                break_t = torch.unsqueeze(break_condition, 1).to(device)
                next_word = next_word.masked_fill(break_t, 0)
                ys = torch.cat([ys, next_word], dim=1)
                next_word = torch.reshape(next_word, (next_word.shape[0],))
                nlls += model._nll_loss(log_prob, next_word)

            break_condition = break_condition | (next_word == end_token)
            if all(break_condition):
                break

        output_smiles_list = [
            tokenizer.untokenize(vocabulary.decode(seq)) for seq in ys.detach().cpu().numpy()
        ]
        input_smiles_list = [
            tokenizer.untokenize(vocabulary.decode(seq)) for seq in src.detach().cpu().numpy()
        ]
        result = (
            input_smiles_list,
            output_smiles_list,
            nlls.detach().cpu().numpy().tolist(),
        )
    return result, encoder_outputs.to(device)


def generate_feature(model, mol, device, output_dim=2000):
    # Generate initial features
    encoder_out = generate_samples(model, mol, beam_size=1000, device=device).squeeze(0).flatten()
    
    # Determine input features
    if isinstance(encoder_out, torch.Tensor):
        in_features = encoder_out.shape[0]
    else:
        in_features = len(encoder_out)
    
    # Initialize feature_adapter if not provided
    
    feature_adapter = nn.Linear(in_features, output_dim).to(device)
    
    # Ensure encoder_out is on the correct device
    if isinstance(encoder_out, np.ndarray):
        encoder_out = torch.from_numpy(encoder_out).float().to(device)
    elif isinstance(encoder_out, torch.Tensor):
        encoder_out = encoder_out.to(device)
    else:
        raise TypeError("encoder_out must be a torch.Tensor or np.ndarray")
    
    # Apply linear layer
    
    standardized_features = feature_adapter(encoder_out)
    
    # Convert to list (move to CPU first)
    if isinstance(standardized_features, torch.Tensor):
        standardized_features = standardized_features.detach().cpu().numpy()
    feature_list = standardized_features.tolist()
    
    return feature_list

# def generate_feature_new_logging(model, mol_list, device):
#     # Convert list of SMILES strings to tensors
#     src_list = []
#     for smiles in mol_list:
#         mol = Chem.MolFromSmiles(smiles)
#         smi_no_iso = remove_isotopes(mol)
#         smiles_std = standardize_smiles(smi_no_iso)
#         src = model.vocabulary.encode(model.tokenizer.tokenize(smiles_std))
#         src = torch.from_numpy(src.astype(np.int64))
#         src_list.append(src)
    
#     print("\nTokenized sequences before padding:")
#     for idx, seq in enumerate(src_list):  
#         print(f"Sequence {idx}: {seq}")

#     # Pad sequences to the same length
#     src_padded = torch.nn.utils.rnn.pad_sequence(
#         src_list, batch_first=True, padding_value=model.vocabulary.pad_token
#     )
#     print("\nPadded sequences:")
#     print(src_padded)
    
#     # Create source mask
#     src_mask = (src_padded == model.vocabulary.pad_token).unsqueeze(-2)
#     print("\nSource mask:")
#     print(src_mask)
    
#     # Move tensors to device
#     src_padded = src_padded.to(device)
#     src_mask = src_mask.to(device)
    
#     # Encode the batch
#     with torch.no_grad():
#         encoder_outputs = model.network.encode(src_padded, src_mask)
#         # encoder_outputs shape: (batch_size, sequence_length, model_dim)
#     print("\nEncoder outputs:")
#     print(encoder_outputs)

#     # Apply mean pooling over the sequence dimension
#     # First, zero out the padding positions
#     src_mask_seq = src_mask.squeeze(1).permute(0, 1)  # Shape: (batch_size, sequence_length)
#     encoder_outputs = encoder_outputs * (~src_mask_seq).unsqueeze(-1)

#     print("\nMasked encoder outputs:")
#     print(encoder_outputs)
    
#     # Sum over the sequence length
#     sum_enc_outputs = encoder_outputs.sum(dim=1)  # Shape: (batch_size, model_dim)
    
#     # Count the number of valid (non-padding) tokens
#     lengths = src_mask_seq.sum(dim=1)  # Shape: (batch_size)
#     lengths = lengths.unsqueeze(-1)  # Shape: (batch_size, 1)
    
#     # Avoid division by zero
#     lengths = lengths.clamp(min=1)
    
#     # Compute mean over the sequence length
#     mean_enc_outputs = sum_enc_outputs / lengths  # Shape: (batch_size, model_dim)
    
#     # No linear layer needed
#     standardized_features = mean_enc_outputs  # Shape: (batch_size, model_dim)
    
#     # Return as tensor (no need to convert to list)
#     return standardized_features


def generate_feature_1(model, mol_list, device):
    # Convert list of SMILES strings to tensors
    src_list = []
    for smiles in mol_list:
        mol = Chem.MolFromSmiles(smiles)
        smi_no_iso = remove_isotopes(mol)
        smiles_std = standardize_smiles(smi_no_iso)
        src = model.vocabulary.encode(model.tokenizer.tokenize(smiles_std))
        src = torch.from_numpy(src.astype(np.int64))
        src_list.append(src)
    
    # Pad sequences to the same length
    src_padded = torch.nn.utils.rnn.pad_sequence(
        src_list, batch_first=True, padding_value=model.vocabulary.pad_token
    )
    
    # Create source mask
    src_mask = (src_padded == model.vocabulary.pad_token).unsqueeze(-2)
    
    # Move tensors to device
    src_padded = src_padded.to(device)
    src_mask = src_mask.to(device)
    
    # Encode the batch
    with torch.no_grad():
        encoder_outputs = model.network.encode(src_padded, src_mask)
        # encoder_outputs shape: (batch_size, sequence_length, model_dim)
    
    # Apply mean pooling over the sequence dimension
    # First, zero out the padding positions
    src_mask_seq = src_mask.squeeze(1).permute(0, 1)  # Shape: (batch_size, sequence_length)
    encoder_outputs = encoder_outputs * (~src_mask_seq).unsqueeze(-1)
    
    # Sum over the sequence length
    sum_enc_outputs = encoder_outputs.sum(dim=1)  # Shape: (batch_size, model_dim)
    
    # Count the number of valid (non-padding) tokens
    lengths = src_mask_seq.sum(dim=1)  # Shape: (batch_size)
    lengths = lengths.unsqueeze(-1)  # Shape: (batch_size, 1)
    
    # Avoid division by zero
    lengths = lengths.clamp(min=1)
    
    # Compute mean over the sequence length
    mean_enc_outputs = sum_enc_outputs / lengths  # Shape: (batch_size, model_dim)
    
    # No linear layer needed
    standardized_features = mean_enc_outputs  # Shape: (batch_size, model_dim)
    
    # Return as tensor (no need to convert to list)
    return standardized_features

def generate_feature_2(model, mol_list, device):
    # Convert list of SMILES strings to tensors
    src_list = []
    for smiles in mol_list:
        mol = Chem.MolFromSmiles(smiles)
        smi_no_iso = remove_isotopes(mol)
        smiles_std = standardize_smiles(smi_no_iso)
        src = model.vocabulary.encode(model.tokenizer.tokenize(smiles_std))
        src = torch.from_numpy(src.astype(np.int64))
        src_list.append(src)
    
    # Pad sequences to the same length
    src_padded = torch.nn.utils.rnn.pad_sequence(
        src_list, batch_first=True, padding_value=model.vocabulary.pad_token
    )
    
    # Create source mask
    src_mask = (src_padded != model.vocabulary.pad_token).unsqueeze(-2)
    
    # Move tensors to device
    src_padded = src_padded.to(device)
    src_mask = src_mask.to(device)
    
    # Encode the batch
    with torch.no_grad():
        encoder_outputs = model.network.encode(src_padded, src_mask)
        # encoder_outputs shape: (batch_size, sequence_length, model_dim)
    
    # Apply mean pooling over the sequence dimension
    # First, zero out the padding positions
    src_mask_seq = src_mask.squeeze(1).permute(0, 1)  # Shape: (batch_size, sequence_length)
    encoder_outputs = encoder_outputs * src_mask_seq.unsqueeze(-1)
    
    # Sum over the sequence length
    sum_enc_outputs = encoder_outputs.sum(dim=1)  # Shape: (batch_size, model_dim)
    
    # Count the number of valid (non-padding) tokens
    lengths = src_mask_seq.sum(dim=1)  # Shape: (batch_size)
    lengths = lengths.unsqueeze(-1)  # Shape: (batch_size, 1)
    
    # Avoid division by zero
    lengths = lengths.clamp(min=1)
    
    # Compute mean over the sequence length
    mean_enc_outputs = sum_enc_outputs / lengths  # Shape: (batch_size, model_dim)
    
    # No linear layer needed
    standardized_features = mean_enc_outputs  # Shape: (batch_size, model_dim)
    
    # Return as tensor (no need to convert to list)
    return standardized_features