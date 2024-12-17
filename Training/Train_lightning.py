import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2" 
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.loggers import NeptuneLogger
import neptune
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, Callback

from Mol_Feature import load_model, generate_feature_2
from Rule_Feature import generate_rule_feature_vector
from Network import ContrastiveModel1, ContrastiveModel2

neptune_logger = NeptuneLogger(
project="mahootiha-maryam/Drug-Discovery-AI",
api_key="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJmNGE0YWJjZi0yMjFhLTQ5Y2YtOGEwZS03OTY3NzA4MDI4YmUifQ==",
)
############################
class ReactionDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.mol1 = self.data['mol1'].values
        self.mol2 = self.data['mol2'].values
        self.rule = self.data['rule_number'].values

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.mol1[index], self.mol2[index], self.rule[index]
#################################
class CombinedReactionDataset(Dataset):
    def __init__(self, pos_dataset, neg_dataset):
        self.pos_dataset = pos_dataset
        self.neg_dataset = neg_dataset
        assert len(self.pos_dataset) == len(self.neg_dataset), "Datasets must have the same length"

    def __len__(self):
        return len(self.pos_dataset)

    def __getitem__(self, idx):
        pos_sample = self.pos_dataset[idx]
        neg_sample = self.neg_dataset[idx]
        return pos_sample, neg_sample
#################################
# Data Module for PyTorch Lightning
class ReactionDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=10, num_workers=8):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers

    def prepare_data(self):
        pass  # No operation needed here

    def setup(self, stage=None):
        # Load datasets unconditionally
        # Training datasets
        neg_pairs_tr = pd.read_csv('neg_goodpairs_tr.csv')
        pos_pairs_tr = pd.read_csv('pos_goodpairs_tr.csv')
        neg_pairs_tr = neg_pairs_tr.sample(n=100_000, random_state=121274)
        pos_pairs_tr = pos_pairs_tr.sample(n=100_000, random_state=121274)
        self.ds_neg_train = ReactionDataset(neg_pairs_tr)
        self.ds_pos_train = ReactionDataset(pos_pairs_tr)
        self.train_dataset = CombinedReactionDataset(self.ds_pos_train, self.ds_neg_train)

        # Validation datasets
        neg_pairs_val = pd.read_csv('neg_goodpairs_val.csv')
        pos_pairs_val = pd.read_csv('pos_goodpairs_val.csv')
        neg_pairs_val = neg_pairs_val.sample(n=10_000, random_state=121274)
        pos_pairs_val = pos_pairs_val.sample(n=10_000, random_state=121274)
        self.ds_neg_val = ReactionDataset(neg_pairs_val)
        self.ds_pos_val = ReactionDataset(pos_pairs_val)
        self.val_dataset = CombinedReactionDataset(self.ds_pos_val, self.ds_neg_val)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size,
                          shuffle=True, num_workers=self.num_workers, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size,
                          shuffle=False, num_workers=self.num_workers, pin_memory=True)
########################################
# Lightning Module for the model
class ContrastiveLightningModule(pl.LightningModule):
    def __init__(self, input_dim_b, input_dim_c, input_dim_r, embedding_dim, hidden_dim,
                 config_path, checkpoint_path, vocabulary_path, total_rules=56, lr=0.001, margin=1.0):
        super().__init__()
        self.save_hyperparameters()
        self.model = ContrastiveModel2(input_dim_b, input_dim_c, input_dim_r, embedding_dim, hidden_dim)
        self.model_molfeatures = load_model(config_path, checkpoint_path, vocabulary_path, device=self.device)
        self.total_rules = total_rules
        self.lr = lr
        self.margin = margin

    def forward(self, inputs_b, inputs_c, inputs_r):
        return self.model(inputs_b, inputs_c, inputs_r)

    def triplet_loss(self, output_pos, output_neg):
        loss = nn.functional.relu(self.margin - output_pos + output_neg)
        return loss.mean()

    def training_step(self, batch, batch_idx):
        (b_pos, c_pos, r_pos), (b_neg, c_neg, r_neg) = batch

        feat_b_pos = generate_feature_2(self.model_molfeatures, b_pos, self.device)
        feat_c_pos = generate_feature_2(self.model_molfeatures, c_pos, self.device)
        feat_r_pos = [generate_rule_feature_vector(r, self.total_rules) for r in r_pos]

        feat_b_neg = generate_feature_2(self.model_molfeatures, b_neg, self.device)
        feat_c_neg = generate_feature_2(self.model_molfeatures, c_neg, self.device)
        feat_r_neg = [generate_rule_feature_vector(r, self.total_rules) for r in r_neg]

        inputs_b_pos = feat_b_pos.to(self.device)
        inputs_c_pos = feat_c_pos.to(self.device)
        inputs_r_pos = torch.stack([torch.tensor(feat, dtype=torch.float32) for feat in feat_r_pos]).to(self.device)

        inputs_b_neg = feat_b_neg.to(self.device)
        inputs_c_neg = feat_c_neg.to(self.device)
        inputs_r_neg = torch.stack([torch.tensor(feat, dtype=torch.float32) for feat in feat_r_neg]).to(self.device)

        output_pos = self(inputs_b_pos, inputs_c_pos, inputs_r_pos)
        output_neg = self(inputs_b_neg, inputs_c_neg, inputs_r_neg)

        loss = self.triplet_loss(output_pos, output_neg)

        self.log('train_loss', loss, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        (b_pos, c_pos, r_pos), (b_neg, c_neg, r_neg) = batch

        feat_b_pos = generate_feature_2(self.model_molfeatures, b_pos, self.device)
        feat_c_pos = generate_feature_2(self.model_molfeatures, c_pos, self.device)
        feat_r_pos = [generate_rule_feature_vector(r, self.total_rules) for r in r_pos]

        feat_b_neg = generate_feature_2(self.model_molfeatures, b_neg, self.device)
        feat_c_neg = generate_feature_2(self.model_molfeatures, c_neg, self.device)
        feat_r_neg = [generate_rule_feature_vector(r, self.total_rules) for r in r_neg]

        inputs_b_pos = feat_b_pos.to(self.device)
        inputs_c_pos = feat_c_pos.to(self.device)
        inputs_r_pos = torch.stack([torch.tensor(feat, dtype=torch.float32) for feat in feat_r_pos]).to(self.device)

        inputs_b_neg = feat_b_neg.to(self.device)
        inputs_c_neg = feat_c_neg.to(self.device)
        inputs_r_neg = torch.stack([torch.tensor(feat, dtype=torch.float32) for feat in feat_r_neg]).to(self.device)

        output_pos = self(inputs_b_pos, inputs_c_pos, inputs_r_pos)
        output_neg = self(inputs_b_neg, inputs_c_neg, inputs_r_neg)

        loss = self.triplet_loss(output_pos, output_neg)
        # run['val_loss'].log(loss)
        self.log('val_loss', loss, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

#########################################
config_path = "/projects/mai/se_mai/users/kvvq085_Mary/config.yml"
checkpoint_path = "/projects/mai/se_mai/users/kvvq085_Mary/weights.ckpt"
vocabulary_path = "/projects/mai/se_mai/users/kvvq085_Mary/vocabulary.pkl"
#########################################
#working
input_dim_b = 256
input_dim_c = 256
input_dim_r = 56
embedding_dim = 100
hidden_dim = 50
num_epochs = 150
#####################################
data_module = ReactionDataModule(batch_size=10, num_workers=8)
######################################
model = ContrastiveLightningModule(
    input_dim_b=input_dim_b,
    input_dim_c=input_dim_c,
    input_dim_r=input_dim_r,
    embedding_dim=embedding_dim,
    hidden_dim=hidden_dim,
    config_path=config_path,
    checkpoint_path=checkpoint_path,
    vocabulary_path=vocabulary_path,
    total_rules=56,
    lr=0.001,
    margin=1.0
)
#######################################
class SaveLossToCSVCallback(Callback):
    def __init__(self, csv_path='loss_log2.csv'):
        super().__init__()
        self.csv_path = csv_path
        # Initialize the CSV file with headers if it doesn't exist
        if not os.path.exists(self.csv_path) and self.is_global_zero:
            df = pd.DataFrame(columns=['epoch', 'train_loss', 'val_loss'])
            df.to_csv(self.csv_path, index=False)

    def on_epoch_end(self, trainer, pl_module):
        if not trainer.is_global_zero:
            return  # Only the main process will write to the CSV

        epoch = trainer.current_epoch
        # Retrieve the latest logged metrics
        train_loss = trainer.callback_metrics.get('train_loss')
        val_loss = trainer.callback_metrics.get('val_loss')

        # Ensure that both losses are available
        if train_loss is not None and val_loss is not None:
            # Create a new DataFrame row
            new_row = pd.DataFrame({
                'epoch': [epoch],
                'train_loss': [train_loss.item()],
                'val_loss': [val_loss.item()]
            })

            # Append the new row to the CSV file
            new_row.to_csv(self.csv_path, mode='a', header=False, index=False)

csv_callback = SaveLossToCSVCallback(csv_path='loss_log_trans_ohe.csv')
######################################
early_stop_callback = EarlyStopping(monitor='val_loss', patience=5, verbose=True, mode='min')
checkpoint_callback = ModelCheckpoint(
    monitor='val_loss',
    dirpath='checkpoints2',
    filename='best_model_trohe_{epoch:02d}_{val_loss:.2f}',
    save_top_k=1,
    mode='min',
)
############################################
# Initialize the trainer

trainer = pl.Trainer(
    max_epochs=num_epochs,
    accelerator='gpu',
    logger=neptune_logger,
    devices=3,  
    callbacks=[early_stop_callback, checkpoint_callback, csv_callback],
    strategy='ddp_find_unused_parameters_true', # Distributed Data Parallel
    check_val_every_n_epoch=10     
)

# Train the model
trainer.fit(model, datamodule=data_module)
