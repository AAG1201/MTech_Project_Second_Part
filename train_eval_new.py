import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader

class AttentionBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(AttentionBlock, self).__init__()
        self.query = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        self.key = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        self.value = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, x):
        batch_size, C, width = x.size()
        
        # Generate projections
        proj_query = self.query(x).permute(0, 2, 1)  # B x W x C
        proj_key = self.key(x)  # B x C x W
        energy = torch.bmm(proj_query, proj_key)  # B x W x W
        attention = self.softmax(energy)  # B x W x W
        
        proj_value = self.value(x)  # B x C x W
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))  # B x C x W
        
        out = self.gamma * out + x
        return out

class FrequencyAwareAttention(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FrequencyAwareAttention, self).__init__()
        self.attention = AttentionBlock(in_channels, out_channels)
        
        # Frequency positional encoding
        self.freq_embedding = nn.Parameter(torch.randn(1, out_channels, 1000))
        
    def forward(self, x):
        # Add frequency positional encoding
        batch_size, channels, width = x.size()
        freq_embed = self.freq_embedding[:, :, :width].expand(batch_size, -1, -1)
        x = x + freq_embed
        
        # Apply attention
        x = self.attention(x)
        return x

class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool1d(2)
        
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        skip = x
        x = self.pool(x)
        return x, skip

class UpBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        """
        Args:
            in_channels: Number of input channels from the previous layer
            skip_channels: Number of channels from the skip connection
            out_channels: Number of output channels
        """
        super(UpBlock, self).__init__()
        # Upsample reduces channels by half
        self.up = nn.ConvTranspose1d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        
        # After concatenation, we'll have (in_channels//2 + skip_channels) channels
        concat_channels = in_channels // 2 + skip_channels
        
        # Now we need to reduce from concat_channels to out_channels
        self.conv1 = nn.Conv1d(concat_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x, skip):
        x = self.up(x)
        
        # Handle cases where dimensions don't match
        diff_H = skip.shape[2] - x.shape[2]
        if diff_H > 0:
            x = F.pad(x, [diff_H // 2, diff_H - diff_H // 2])
        elif diff_H < 0:
            skip = F.pad(skip, [-diff_H // 2, -diff_H + diff_H // 2])
        
        # Concatenate along the channel dimension
        x = torch.cat([x, skip], dim=1)
        
        # Apply convolutions
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x

class ComplexDFTUNet(nn.Module):
    def __init__(self, dft_size=1000, max_sparsity=500):
        super(ComplexDFTUNet, self).__init__()
        self.dft_size = dft_size
        self.max_sparsity = max_sparsity
        
        # Input is 1000 (real and imaginary parts of 500 complex numbers)
        self.inc = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True)
        )
        
        # Downsampling path - track channel dimensions
        self.down1 = DownBlock(64, 128)      # 64 -> 128
        self.down2 = DownBlock(128, 256)     # 128 -> 256
        self.down3 = DownBlock(256, 512)     # 256 -> 512
        self.down4 = DownBlock(512, 1024)    # 512 -> 1024
        
        # Frequency-aware attention at the bottleneck
        self.freq_attention = FrequencyAwareAttention(1024, 1024)
        
        # Upsampling path - specify all three channel dimensions clearly
        # (in_channels, skip_channels, out_channels)
        self.up1 = UpBlock(1024, 1024, 512)  # 1024 + 1024/2 concat -> 512
        self.up2 = UpBlock(512, 512, 256)    # 512 + 512/2 concat -> 256
        self.up3 = UpBlock(256, 256, 128)    # 256 + 256/2 concat -> 128
        self.up4 = UpBlock(128, 128, 64)      # 128 + 64/2 concat -> 64
        
        # Output layer for DFT estimation
        self.outc = nn.Conv1d(64, 1, kernel_size=1)
        
        # Sparsity estimation branch
        self.sparsity_branch = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()  # Output will be scaled between 0 and 1
        )
        
    def forward(self, x):
        # Reshape input to [batch, channels, sequence_length]
        batch_size = x.size(0)
        x = x.view(batch_size, 1, -1)
        
        # Encoder path
        x1 = self.inc(x)
        x2, skip1 = self.down1(x1)
        x3, skip2 = self.down2(x2)
        x4, skip3 = self.down3(x3)
        x5, skip4 = self.down4(x4)
        
        # Apply frequency-aware attention at the bottleneck
        x5 = self.freq_attention(x5)
        
        # Sparsity estimation
        sparsity = self.sparsity_branch(x5)
        sparsity = sparsity * self.max_sparsity  # Scale to [0, max_sparsity]
        
        # Decoder path
        x = self.up1(x5, skip4)
        x = self.up2(x, skip3)
        x = self.up3(x, skip2)
        x = self.up4(x, skip1)
        
        # Output for DFT
        dft_output = self.outc(x)
        dft_output = dft_output.view(batch_size, -1)  # Reshape to [batch, sequence_length]
        
        return dft_output, sparsity

class ComplexDFTDataset(Dataset):
    def __init__(self, inputs, targets_dft, targets_sparsity=None, max_sparsity=500):
        """
        Args:
            inputs: Array of input signals [N, 1000] (500 real + 500 imaginary)
            targets_dft: Array of target DFT outputs [N, 1000] (500 real + 500 imaginary)
            targets_sparsity: Array of target sparsity values [N]
            max_sparsity: Maximum possible sparsity value
        """
        self.inputs = torch.tensor(inputs, dtype=torch.float32)
        self.targets_dft = torch.tensor(targets_dft, dtype=torch.float32)
        
        if targets_sparsity is not None:
            self.targets_sparsity = torch.tensor(targets_sparsity, dtype=torch.float32).view(-1, 1)
        else:
            self.targets_sparsity = None
            
        self.max_sparsity = max_sparsity
        
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        x = self.inputs[idx]
        y_dft = self.targets_dft[idx]
        
        if self.targets_sparsity is not None:
            y_sparsity = self.targets_sparsity[idx]
            return x, y_dft, y_sparsity
        else:
            return x, y_dft


def train_complex_dft_unet(model, train_loader, val_loader=None, epochs=100, lr=0.001, 
                          plot_loss=True, checkpoint_dir='checkpoints', 
                          checkpoint_freq=5, resume_from=None):
    """
    Train the Complex DFT U-Net model with checkpointing
    
    Args:
        model: The model to train
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data (optional)
        epochs: Number of epochs to train
        lr: Learning rate
        plot_loss: Whether to plot loss curves at the end
        checkpoint_dir: Directory to save checkpoints
        checkpoint_freq: How often to save checkpoints (in epochs)
        resume_from: Path to checkpoint to resume training from (optional)
    
    Returns:
        Dictionary containing loss history
    """
    import os
    import matplotlib.pyplot as plt
    from datetime import datetime
    
    # Create checkpoint directory if it doesn't exist
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Loss and optimizer
    criterion_dft = nn.MSELoss()
    criterion_sparsity = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5, verbose=True)
    
    # Training history
    train_losses = []
    val_losses = []
    dft_losses = []
    sparsity_losses = []
    
    # Starting epoch
    start_epoch = 0
    
    # Resume from checkpoint if specified
    if resume_from is not None and os.path.isfile(resume_from):
        print(f"Loading checkpoint from {resume_from}")
        checkpoint = torch.load(resume_from, map_location=device)
        
        # Load model state
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer state
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load scheduler state
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Get the epoch we're resuming from
        start_epoch = checkpoint['epoch'] + 1
        
        # Load loss history
        train_losses = checkpoint['train_losses']
        val_losses = checkpoint['val_losses']
        dft_losses = checkpoint['dft_losses']
        sparsity_losses = checkpoint['sparsity_losses']
        
        print(f"Resuming training from epoch {start_epoch}")
    
    # Training loop
    for epoch in range(start_epoch, epochs):
        model.train()
        epoch_loss = 0
        epoch_dft_loss = 0
        epoch_sparsity_loss = 0
        
        for batch_idx, (inputs, targets_dft, targets_sparsity) in enumerate(train_loader):
            inputs = inputs.to(device)
            targets_dft = targets_dft.to(device)
            targets_sparsity = targets_sparsity.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs_dft, outputs_sparsity = model(inputs)
            
            # Calculate losses
            loss_dft = criterion_dft(outputs_dft, targets_dft)
            loss_sparsity = criterion_sparsity(outputs_sparsity, targets_sparsity)
            
            # Combined loss (you can adjust the weights)
            loss = loss_dft + 0.1 * loss_sparsity
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            epoch_dft_loss += loss_dft.item()
            epoch_sparsity_loss += loss_sparsity.item()
            
            # Print progress
            if (batch_idx + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Batch [{batch_idx+1}/{len(train_loader)}], '
                      f'Loss: {loss.item():.4f}, DFT Loss: {loss_dft.item():.4f}, '
                      f'Sparsity Loss: {loss_sparsity.item():.4f}')
        
        avg_train_loss = epoch_loss / len(train_loader)
        avg_dft_loss = epoch_dft_loss / len(train_loader)
        avg_sparsity_loss = epoch_sparsity_loss / len(train_loader)
        
        train_losses.append(avg_train_loss)
        dft_losses.append(avg_dft_loss)
        sparsity_losses.append(avg_sparsity_loss)
        
        # Validation
        if val_loader is not None:
            model.eval()
            val_loss = 0
            
            with torch.no_grad():
                for inputs, targets_dft, targets_sparsity in val_loader:
                    inputs = inputs.to(device)
                    targets_dft = targets_dft.to(device)
                    targets_sparsity = targets_sparsity.to(device)
                    
                    outputs_dft, outputs_sparsity = model(inputs)
                    
                    loss_dft = criterion_dft(outputs_dft, targets_dft)
                    loss_sparsity = criterion_sparsity(outputs_sparsity, targets_sparsity)
                    loss = loss_dft + 0.1 * loss_sparsity
                    
                    val_loss += loss.item()
            
            avg_val_loss = val_loss / len(val_loader)
            val_losses.append(avg_val_loss)
            
            # Update learning rate
            scheduler.step(avg_val_loss)
            
            print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.4f}, '
                  f'Val Loss: {avg_val_loss:.4f}, DFT Loss: {avg_dft_loss:.4f}, '
                  f'Sparsity Loss: {avg_sparsity_loss:.4f}')
        else:
            # Update learning rate based on training loss
            scheduler.step(avg_train_loss)
            
            print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.4f}, '
                  f'DFT Loss: {avg_dft_loss:.4f}, Sparsity Loss: {avg_sparsity_loss:.4f}')
        
        # Save checkpoint
        if (epoch + 1) % checkpoint_freq == 0 or (epoch + 1) == epochs:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch+1}_{timestamp}.pt')
            
            # Create checkpoint
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_losses': train_losses,
                'val_losses': val_losses,
                'dft_losses': dft_losses,
                'sparsity_losses': sparsity_losses
            }
            
            # Save checkpoint
            torch.save(checkpoint, checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")
            
            # Also save a 'latest' checkpoint for easy resuming
            latest_path = os.path.join(checkpoint_dir, 'checkpoint_latest.pt')
            torch.save(checkpoint, latest_path)
            print(f"Latest checkpoint updated at {latest_path}")
    
    # Plot training history if requested
    if plot_loss:
        plt.figure(figsize=(10, 6))
        plt.plot(range(start_epoch + 1, epochs+1), train_losses[start_epoch:], 'b-', label='Training Loss')
        if val_losses:
            plt.plot(range(start_epoch + 1, epochs+1), val_losses[start_epoch:], 'r-', label='Validation Loss')
        plt.title('Model Loss During Training')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(checkpoint_dir, "training_loss.png"), dpi=300, bbox_inches='tight')
        plt.show()
        
        # Plot individual loss components
        plt.figure(figsize=(10, 6))
        plt.plot(range(start_epoch + 1, epochs+1), dft_losses[start_epoch:], 'g-', label='DFT Loss')
        plt.plot(range(start_epoch + 1, epochs+1), sparsity_losses[start_epoch:], 'm-', label='Sparsity Loss')
        plt.title('Loss Components During Training')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(checkpoint_dir, "loss_components.png"), dpi=300, bbox_inches='tight')
        plt.show()
    
    # Return all loss histories for further analysis
    loss_history = {
        'train_loss': train_losses,
        'val_loss': val_losses,
        'dft_loss': dft_losses,
        'sparsity_loss': sparsity_losses
    }
    
    return loss_history

def find_latest_checkpoint(checkpoint_dir='checkpoints'):
    """
    Find the latest checkpoint in the given directory
    
    Args:
        checkpoint_dir: Directory containing checkpoints
        
    Returns:
        Path to the latest checkpoint file, or None if no checkpoints found
    """
    import os
    import glob
    
    # Check if latest checkpoint exists
    latest_path = os.path.join(checkpoint_dir, 'checkpoint_latest.pt')
    if os.path.exists(latest_path):
        return latest_path
    
    # Otherwise, find the checkpoint with the highest epoch number
    checkpoints = glob.glob(os.path.join(checkpoint_dir, 'checkpoint_epoch_*.pt'))
    
    if not checkpoints:
        return None
    
    # Sort by epoch number and timestamp
    latest_checkpoint = max(checkpoints, key=os.path.getctime)
    return latest_checkpoint

def save_model(model, path):
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")


def load_model(model, path, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    return model


def evaluate_model(model, test_loader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    dft_losses = []
    sparsity_losses = []
    
    with torch.no_grad():
        for inputs, targets_dft, targets_sparsity in test_loader:
            inputs = inputs.to(device)
            targets_dft = targets_dft.to(device)
            targets_sparsity = targets_sparsity.to(device)
            
            outputs_dft, outputs_sparsity = model(inputs)
            
            dft_loss = F.mse_loss(outputs_dft, targets_dft).item()
            sparsity_loss = F.mse_loss(outputs_sparsity, targets_sparsity).item()
            
            dft_losses.append(dft_loss)
            sparsity_losses.append(sparsity_loss)
    
    avg_dft_loss = sum(dft_losses) / len(dft_losses)
    avg_sparsity_loss = sum(sparsity_losses) / len(sparsity_losses)
    
    print(f"Evaluation Results:")
    print(f"Average DFT Loss: {avg_dft_loss:.6f}")
    print(f"Average Sparsity Loss: {avg_sparsity_loss:.6f}")
    
    return avg_dft_loss, avg_sparsity_loss


def debug_model(model, train_loader):
    """Print input/output dimensions of each layer to help diagnose dimension mismatches"""
    print("Running debug mode to check dimensions...")
    device = torch.device('cpu')  # Use CPU for debugging
    model = model.to(device)
    model.eval()
    
    # Get a sample batch
    sample_batch = next(iter(train_loader))
    inputs = sample_batch[0].to(device)
    print(f"Input shape: {inputs.shape}")
    
    # Trace through the model
    try:
        batch_size = inputs.size(0)
        x = inputs.view(batch_size, 1, -1)
        print(f"Reshaped input: {x.shape}")
        
        # Encoder
        x1 = model.inc(x)
        print(f"After inc: {x1.shape}")
        
        x2, skip1 = model.down1(x1)
        print(f"After down1: {x2.shape}, skip1: {skip1.shape}")
        
        x3, skip2 = model.down2(x2)
        print(f"After down2: {x3.shape}, skip2: {skip2.shape}")
        
        x4, skip3 = model.down3(x3)
        print(f"After down3: {x4.shape}, skip3: {skip3.shape}")
        
        x5, skip4 = model.down4(x4)
        print(f"After down4: {x5.shape}, skip4: {skip4.shape}")
        
        # Attention
        x5 = model.freq_attention(x5)
        print(f"After attention: {x5.shape}")
        
        # Sparsity
        sparsity = model.sparsity_branch(x5)
        print(f"Sparsity output: {sparsity.shape}")
        
        # Decoder
        x = model.up1(x5, skip4)
        print(f"After up1: {x.shape}")
        
        x = model.up2(x, skip3)
        print(f"After up2: {x.shape}")
        
        x = model.up3(x, skip2)
        print(f"After up3: {x.shape}")
        
        x = model.up4(x, skip1)
        print(f"After up4: {x.shape}")
        
        # Output
        dft_output = model.outc(x)
        print(f"After outc: {dft_output.shape}")
        
        dft_output = dft_output.view(batch_size, -1)
        print(f"Final DFT output: {dft_output.shape}")
        
        print("Debug successful - all dimensions match correctly")
        
    except Exception as e:
        print(f"Debug failed at: {e}")




def predict_with_model(model, input_data, device=None):
    """
    Use the model to predict DFT output and sparsity
    
    Args:
        model: The trained model
        input_data: Input data (can be numpy array or torch tensor)
        device: Device to run prediction on (None for auto-detection)
        
    Returns:
        tuple: (dft_output, predicted_sparsity)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Convert input to tensor if it's not already
    if isinstance(input_data, np.ndarray):
        input_tensor = torch.FloatTensor(input_data)
    else:
        input_tensor = input_data
    
    # Make sure input is on the correct device
    input_tensor = input_tensor.to(device)
    
    # Handle both single samples and batches
    is_single_sample = len(input_tensor.shape) == 1
    if is_single_sample:
        input_tensor = input_tensor.unsqueeze(0)  # Add batch dimension
    
    # Set model to evaluation mode and move to device
    model = model.to(device)
    model.eval()
    
    # Perform prediction
    with torch.no_grad():
        dft_output, sparsity = model(input_tensor)
    
    # Return single sample or batch depending on input
    if is_single_sample:
        return dft_output.squeeze(0).cpu().numpy(), sparsity.item()  # Fixed variable name
    else:
        return dft_output.cpu().numpy(), sparsity.cpu().numpy()  # Fixed variable name