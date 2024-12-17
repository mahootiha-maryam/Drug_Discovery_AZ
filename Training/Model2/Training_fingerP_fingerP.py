import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn as nn
import pandas as pd
import neptune

run = neptune.init_run(
    project="mahootiha-maryam/Drug-Discovery-AI",
    api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJmNGE0YWJjZi0yMjFhLTQ5Y2YtOGEwZS03OTY3NzA4MDI4YmUifQ==",
)

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
    
train_ds_pos = torch.load("positive_train_dataset_FPFP.pt")
train_ds_neg = torch.load("negative_train_dataset_FPFP.pt")
val_ds_pos = torch.load("positive_val_dataset_FPFP.pt")
val_ds_neg = torch.load("negative_val_dataset_FPFP.pt")
test_ds_pos = torch.load("positive_test_dataset_FPFP.pt")
test_ds_neg = torch.load("negative_test_dataset_FPFP.pt")

val_pos_dl = DataLoader(val_ds_pos, batch_size=32, num_workers=4, shuffle=False, drop_last=True)
val_neg_dl = DataLoader(val_ds_neg, batch_size=32, num_workers=4, shuffle=False, drop_last=True)
test_pos_dl = DataLoader(test_ds_pos, batch_size=32, num_workers=4, shuffle=False, drop_last=True)
test_neg_dl = DataLoader(test_ds_neg, batch_size=32, num_workers=4, shuffle=False, drop_last=True)

train_pos_dl = DataLoader(train_ds_pos, batch_size=32, num_workers=4, shuffle=True, drop_last=True)
train_neg_dl = DataLoader(train_ds_neg, batch_size=32, num_workers=4, shuffle=True, drop_last=True)


class ContrastiveModel1(nn.Module):
    def __init__(self, input_dim_b, input_dim_c, input_dim_r, embedding_dim, hidden_dim, intermediate_dim=512):
        super(ContrastiveModel1, self).__init__()
        
        # Two-stage embedding layers for molecule B
        self.embedding_b1 = nn.Linear(input_dim_b, intermediate_dim)
        self.embedding_b2 = nn.Linear(intermediate_dim, embedding_dim)
        
        # Two-stage embedding layers for molecule C
        self.embedding_c1 = nn.Linear(input_dim_c, intermediate_dim)
        self.embedding_c2 = nn.Linear(intermediate_dim, embedding_dim)
        
        # Embedding layer for rule R (single stage since rule vector may be smaller)
        self.embedding_r = nn.Linear(input_dim_r, embedding_dim)
        
        # Fully connected layers
        self.fc1 = nn.Linear(embedding_dim * 3, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        
        # Activation functions
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, b, c, r):
        # Two-stage embedding for molecule B
        emb_b = self.relu(self.embedding_b1(b))
        emb_b = self.embedding_b2(emb_b)
        
        # Two-stage embedding for molecule C
        emb_c = self.relu(self.embedding_c1(c))
        emb_c = self.embedding_c2(emb_c)
        
        # Single-stage embedding for rule R
        emb_r = self.embedding_r(r)
        
        # Concatenate the embeddings
        combined = torch.cat((emb_b, emb_c, emb_r), dim=1)
        
        # Pass through fully connected layers
        hidden = self.relu(self.fc1(combined))
        output = self.fc2(hidden)
        output = self.sigmoid(output)
        
        return output
    
def triplet_loss(output_pos, output_neg, margin=1.0):
    loss = nn.functional.relu(margin - output_pos + output_neg)
    return loss.mean()

input_dim_b = 1024
input_dim_c = 1024
input_dim_r = 2048
embedding_dim = 100
hidden_dim = 50
num_epochs = 100

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0, verbose=False):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_model = None

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.save_checkpoint(model)
            self.counter = 0

    def save_checkpoint(self, model):
        self.best_model = model.state_dict()

train_losses = []
val_losses = []
epochs_list = []

# Initialize model and optimizer
model = ContrastiveModel1(input_dim_b, input_dim_c, input_dim_r, embedding_dim, hidden_dim)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Initialize early stopping
early_stopping = EarlyStopping(patience=5, verbose=True)

validation_frequency = 10
best_val_loss = float('inf')
for epoch in tqdm(range(num_epochs), desc="Epochs"):
    model.train()
    train_loss = 0.0

    for batch_pos, batch_neg in tqdm(zip(train_pos_dl, train_neg_dl)):
        b_pos, c_pos, r_pos, labels_pos = batch_pos
        b_neg, c_neg, r_neg, labels_neg = batch_neg
        
        output_pos = model(b_pos, c_pos, r_pos)
        output_neg = model(b_neg, c_neg, r_neg)

        loss = triplet_loss(output_pos, output_neg)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()

    train_loss /= len(train_pos_dl)
    run['train_loss'].log(train_loss)
    train_losses.append(train_loss)
    epochs_list.append(epoch)

    # Validation loop
    if (epoch + 1) % validation_frequency == 0:
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_pos, batch_neg in zip(val_pos_dl, val_neg_dl):
                b_pos, c_pos, r_pos, labels_pos = batch_pos
                b_neg, c_neg, r_neg, labels_neg = batch_neg

                output_pos = model(b_pos, c_pos, r_pos)
                output_neg = model(b_neg, c_neg, r_neg)

                loss = triplet_loss(output_pos, output_neg)
                val_loss += loss.item()

        val_loss /= len(val_pos_dl)
        run['Validation_loss'].log(val_loss)
        val_losses.append(val_loss)
        
        # Check if the current validation loss is better than the best validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # Save the best model
            torch.save(model.state_dict(), 'best_model_fpfp.pt')

        
        # Early stopping
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping triggered")
            # Load best model
            model.load_state_dict(early_stopping.best_model)
            break


train_losses = pd.DataFrame(train_losses, columns=['loss'])
val_losses = pd.DataFrame(val_losses, columns=['loss'])

train_losses.to_csv("train_losses_fpfp.csv")
val_losses.to_csv("val_losses_fpfp.csv")
