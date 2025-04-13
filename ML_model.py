import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils.data import Dataset, DataLoader, TensorDataset


    
# class ResidualBlock(nn.Module):
#     def __init__(self, in_features, hidden_expansion=1.5):
#         super().__init__()
#         hidden_features = int(in_features * hidden_expansion)
#         self.layer = nn.Sequential(
#             nn.Linear(in_features, hidden_features),
#             nn.BatchNorm1d(hidden_features),
#             nn.LeakyReLU(0.1),
#             nn.Linear(hidden_features, in_features),
#             nn.BatchNorm1d(in_features)
#         )
    
#     def forward(self, x):
#         return x + self.layer(x)  # Residual (skip) connection

# class DenseResidualBlock(nn.Module):
#     def __init__(self, in_features, num_layers=3, hidden_expansion=1.5):
#         super().__init__()
#         self.blocks = nn.ModuleList([ResidualBlock(in_features, hidden_expansion) for _ in range(num_layers)])
#         self.norm = nn.LayerNorm(in_features)
    
#     def forward(self, x):
#         for block in self.blocks:
#             x = block(x)
#         return self.norm(x)

# class ASPADEModel(nn.Module):
#     def __init__(self, input_size, output_size, max_sparsity=4000):
#         super(ASPADEModel, self).__init__()
        
#         # Encoder with increased capacity
#         self.fc1 = nn.Linear(input_size, 2048)
#         self.bn1 = nn.BatchNorm1d(2048)
#         self.dense_block1 = DenseResidualBlock(2048, num_layers=3)
        
#         self.fc2 = nn.Linear(2048, 1024)
#         self.bn2 = nn.BatchNorm1d(1024)
#         self.dense_block2 = DenseResidualBlock(1024, num_layers=3)
        
#         self.fc3 = nn.Linear(1024, 512)
#         self.bn3 = nn.BatchNorm1d(512)
#         self.dense_block3 = DenseResidualBlock(512, num_layers=2)
        
#         # Estimate branch (complex values output)
#         self.fc_est1 = nn.Linear(512, 1024)
#         self.bn_est = nn.BatchNorm1d(1024)
#         self.fc_est2 = nn.Linear(1024, 512)
#         self.fc_est3 = nn.Linear(512, output_size)
        
#         # Sparsity classification branch - increased capacity
#         self.fc_spar1 = nn.Linear(512, 1024)
#         self.bn_spar = nn.BatchNorm1d(1024)
#         self.fc_spar2 = nn.Linear(1024, 512)
#         self.fc_spar3 = nn.Linear(512, max_sparsity + 1)  # Classification over [0, max_sparsity]
        
#         self.activation = nn.LeakyReLU(0.1)
#         self.dropout = nn.Dropout(0.4)  # Higher dropout for better regularization
#         self.output_activation = nn.Tanh()  # Optional: Use if estimate should be in [-1,1] range
        
#     def forward(self, x):
#         # Shared encoder
#         x = self.activation(self.bn1(self.fc1(x)))
#         x = self.dense_block1(x)
#         x = self.dropout(x)
        
#         x = self.activation(self.bn2(self.fc2(x)))
#         x = self.dense_block2(x)
#         x = self.dropout(x)
        
#         x = self.activation(self.bn3(self.fc3(x)))
#         x = self.dense_block3(x)
#         x = self.dropout(x)
        
#         # Complex estimate output with deeper network
#         est = self.activation(self.bn_est(self.fc_est1(x)))
#         est = self.dropout(est)
#         est = self.activation(self.fc_est2(est))
#         estimate = self.fc_est3(est)
#         # Uncomment if you need bounded output
#         # estimate = self.output_activation(estimate)
        
#         # Sparsity classification output (logits) with deeper network
#         spar = self.activation(self.bn_spar(self.fc_spar1(x)))
#         spar = self.dropout(spar)
#         spar = self.activation(self.fc_spar2(spar))
#         sparsity_logits = self.fc_spar3(spar)
        
#         return estimate, sparsity_logits

# class ASPADEDataset(Dataset):
#     def __init__(self, inputs, targets_estimates, targets_sparsity):
#         self.inputs = torch.FloatTensor(inputs)
#         self.targets_estimates = torch.FloatTensor(targets_estimates)
#         self.targets_sparsity = torch.LongTensor(targets_sparsity)
    
#     def __len__(self):
#         return len(self.inputs)
    
#     def __getitem__(self, idx):
#         return self.inputs[idx], self.targets_estimates[idx], self.targets_sparsity[idx]



# import torch
# import torch.nn as nn
# from torch.utils.data import Dataset

# class ResidualBlock(nn.Module):
#     def __init__(self, in_features, hidden_expansion=1.2):
#         super().__init__()
#         hidden_features = int(in_features * hidden_expansion)
#         self.layer = nn.Sequential(
#             nn.Linear(in_features, hidden_features),
#             nn.BatchNorm1d(hidden_features),
#             nn.LeakyReLU(0.1),
#             nn.Linear(hidden_features, in_features),
#             nn.BatchNorm1d(in_features)
#         )
    
#     def forward(self, x):
#         return x + self.layer(x)  # Residual (skip) connection

# class DenseResidualBlock(nn.Module):
#     def __init__(self, in_features, num_layers=3, hidden_expansion=1.2):
#         super().__init__()
#         self.blocks = nn.ModuleList([ResidualBlock(in_features, hidden_expansion) for _ in range(num_layers)])
#         self.norm = nn.LayerNorm(in_features)
    
#     def forward(self, x):
#         for block in self.blocks:
#             x = block(x)
#         return self.norm(x)

# class ASPADEModel(nn.Module):
#     def __init__(self, input_size, output_size, max_sparsity):
#         super(ASPADEModel, self).__init__()
#         latent_dim = 1024

#         # Encoder
#         self.fc1 = nn.Linear(input_size, latent_dim)
#         self.bn1 = nn.BatchNorm1d(latent_dim)
#         self.dense_block1 = DenseResidualBlock(latent_dim, num_layers=3)
        
#         self.fc2 = nn.Linear(latent_dim, latent_dim // 2)
#         self.bn2 = nn.BatchNorm1d(latent_dim // 2)
#         self.dense_block2 = DenseResidualBlock(latent_dim // 2, num_layers=3)
        
#         self.fc3 = nn.Linear(latent_dim // 2, latent_dim // 4)
#         self.bn3 = nn.BatchNorm1d(latent_dim // 4)
#         self.dense_block3 = DenseResidualBlock(latent_dim // 4, num_layers=2)
        
#         # Estimate branch
#         self.fc_est1 = nn.Linear(latent_dim // 4, latent_dim // 2)
#         self.bn_est = nn.BatchNorm1d(latent_dim // 2)
#         self.fc_est2 = nn.Linear(latent_dim // 2, output_size)
        
#         # Sparsity classification branch
#         self.fc_spar1 = nn.Linear(latent_dim // 4, latent_dim // 2)
#         self.bn_spar = nn.BatchNorm1d(latent_dim // 2)
#         self.fc_spar2 = nn.Linear(latent_dim // 2, max_sparsity + 1)  # Classification output
        
#         self.activation = nn.LeakyReLU(0.1)
#         self.dropout = nn.Dropout(0.3)  # Regularization
        
#     def forward(self, x):
#         # Shared encoder
#         x = self.activation(self.bn1(self.fc1(x)))
#         x = self.dense_block1(x)
#         x = self.dropout(x)
        
#         x = self.activation(self.bn2(self.fc2(x)))
#         x = self.dense_block2(x)
#         x = self.dropout(x)
        
#         x = self.activation(self.bn3(self.fc3(x)))
#         x = self.dense_block3(x)
#         x = self.dropout(x)
        
#         # Estimate output
#         est = self.activation(self.bn_est(self.fc_est1(x)))
#         est = self.dropout(est)
#         estimate = self.fc_est2(est)
        
#         # Sparsity classification output
#         spar = self.activation(self.bn_spar(self.fc_spar1(x)))
#         spar = self.dropout(spar)
#         sparsity_logits = self.fc_spar2(spar)
        
#         return estimate, sparsity_logits

# class ASPADEDataset(Dataset):
#     def __init__(self, inputs, targets_estimates, targets_sparsity):
#         self.inputs = torch.FloatTensor(inputs)
#         self.targets_estimates = torch.FloatTensor(targets_estimates)
#         self.targets_sparsity = torch.LongTensor(targets_sparsity)
    
#     def __len__(self):
#         return len(self.inputs)
    
#     def __getitem__(self, idx):
#         return self.inputs[idx], self.targets_estimates[idx], self.targets_sparsity[idx]

class ResidualBlock(nn.Module):
    def __init__(self, in_features, hidden_expansion=1.2):
        super().__init__()
        hidden_features = int(in_features * hidden_expansion)
        self.layer = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.BatchNorm1d(hidden_features),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_features, in_features),
            nn.BatchNorm1d(in_features)
        )
    
    def forward(self, x):
        return x + self.layer(x)  # Residual (skip) connection

class DenseResidualBlock(nn.Module):
    def __init__(self, in_features, num_layers=3, hidden_expansion=1.2):
        super().__init__()
        self.blocks = nn.ModuleList([ResidualBlock(in_features, hidden_expansion) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(in_features)
    
    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return self.norm(x)

class ASPADEModel(nn.Module):
    def __init__(self, input_size=4000, output_size=4000, max_sparsity=4000):
        super(ASPADEModel, self).__init__()
        
        # Encoder with dimensions suited for 4000x1 input
        self.fc1 = nn.Linear(input_size, 2048)
        self.bn1 = nn.BatchNorm1d(2048)
        self.dense_block1 = DenseResidualBlock(2048, num_layers=3)
        
        self.fc2 = nn.Linear(2048, 1024)
        self.bn2 = nn.BatchNorm1d(1024)
        self.dense_block2 = DenseResidualBlock(1024, num_layers=3)
        
        self.fc3 = nn.Linear(1024, 512)
        self.bn3 = nn.BatchNorm1d(512)
        self.dense_block3 = DenseResidualBlock(512, num_layers=2)
        
        # Estimate branch for 4000x1 output vector
        self.fc_est1 = nn.Linear(512, 1024)
        self.bn_est = nn.BatchNorm1d(1024)
        self.fc_est2 = nn.Linear(1024, 2048)
        self.fc_est3 = nn.Linear(2048, output_size)
        
        # Sparsity classification branch - for single integer output
        self.fc_spar1 = nn.Linear(512, 512)
        self.bn_spar = nn.BatchNorm1d(512)
        self.fc_spar2 = nn.Linear(512, 256)
        self.fc_spar3 = nn.Linear(256, max_sparsity + 1)  # Classification over [0, max_sparsity]
        
        self.activation = nn.LeakyReLU(0.1)
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        # Ensure input is the right shape for BatchNorm
        if len(x.shape) == 2:
            # Shared encoder
            x = self.activation(self.bn1(self.fc1(x)))
            x = self.dense_block1(x)
            x = self.dropout(x)
            
            x = self.activation(self.bn2(self.fc2(x)))
            x = self.dense_block2(x)
            x = self.dropout(x)
            
            x = self.activation(self.bn3(self.fc3(x)))
            x = self.dense_block3(x)
            x = self.dropout(x)
            
            # Output feature vector (4000x1)
            est = self.activation(self.bn_est(self.fc_est1(x)))
            est = self.dropout(est)
            est = self.activation(self.fc_est2(est))
            estimate = self.fc_est3(est)
            
            # Sparsity integer classification output
            spar = self.activation(self.bn_spar(self.fc_spar1(x)))
            spar = self.dropout(spar)
            spar = self.activation(self.fc_spar2(spar))
            sparsity_logits = self.fc_spar3(spar)
            
            return estimate, sparsity_logits
        else:
            raise ValueError(f"Expected input of shape [batch_size, 4000], got {x.shape}")

class ASPADEDataset(Dataset):
    def __init__(self, inputs, targets_estimates, targets_sparsity):
        """
        Args:
            inputs: Input feature vectors [N, 4000]
            targets_estimates: Target feature vectors [N, 4000]
            targets_sparsity: Target sparsity integers [N]
        """
        self.inputs = torch.FloatTensor(inputs)
        self.targets_estimates = torch.FloatTensor(targets_estimates)
        self.targets_sparsity = torch.LongTensor(targets_sparsity)
    
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        return self.inputs[idx], self.targets_estimates[idx], self.targets_sparsity[idx]