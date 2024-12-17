import torch.nn as nn
import torch

class ContrastiveModel1(nn.Module):
    def __init__(self, input_dim_b, input_dim_c, input_dim_r, embedding_dim, hidden_dim, intermediate_dim=1000):
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
########################################   
class ContrastiveModel2(nn.Module):
    def __init__(self, input_dim_b, input_dim_c, input_dim_r, embedding_dim, hidden_dim):
        super(ContrastiveModel2, self).__init__()
        
        # Two-stage embedding layers for molecule B
        self.embedding_b1 = nn.Linear(input_dim_b, embedding_dim)
        
        # Two-stage embedding layers for molecule C
        self.embedding_c1 = nn.Linear(input_dim_c, embedding_dim)
        
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
        emb_b = self.embedding_b1(b)
        
        # Two-stage embedding for molecule C
        emb_c = self.embedding_c1(c)
        
        # Single-stage embedding for rule R
        emb_r = self.embedding_r(r)
        
        # Concatenate the embeddings
        combined = torch.cat((emb_b, emb_c, emb_r), dim=1)
        
        # Pass through fully connected layers
        hidden = self.relu(self.fc1(combined))
        output = self.fc2(hidden)
        output = self.sigmoid(output)
        
        return output