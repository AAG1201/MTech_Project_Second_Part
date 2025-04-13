import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt

class ComplexDFTModel(nn.Module):
    def __init__(self, dft_size=2000, max_sparsity=1000):
        super(ComplexDFTModel, self).__init__()
        
        # For 2000-point complex DFT (represented as 4000 real values)
        self.input_size = dft_size * 2
        self.output_size = dft_size * 2
        
        # Encoder part
        self.encoder = nn.Sequential(
            nn.Linear(self.input_size, 2048),
            nn.BatchNorm1d(2048),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),
            
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),
            
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.1),
        )
        
        # Decoder for complex DFT output
        self.decoder = nn.Sequential(
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),
            
            nn.Linear(1024, 2048),
            nn.BatchNorm1d(2048),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),
            
            nn.Linear(2048, self.output_size),
        )
        
        # Sparsity prediction branch
        self.sparsity_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),
            nn.Linear(256, max_sparsity + 1)
        )
        
    def forward(self, x):
        # Encode
        features = self.encoder(x)
        
        # Decode to get complex DFT (real and imaginary parts)
        dft_output = self.decoder(features)
        
        # Predict sparsity
        sparsity_logits = self.sparsity_head(features)
        
        return dft_output, sparsity_logits
    
    def complex_representation(self, x):
        """Helper method to convert the real 4000×1 vector back to 2000 complex values"""
        dft_output, _ = self.forward(x)
        batch_size = dft_output.shape[0]
        complex_output = torch.complex(
            dft_output[:, :self.output_size//2], 
            dft_output[:, self.output_size//2:]
        )
        return complex_output.view(batch_size, -1)


class ComplexDFTAttentionModel(nn.Module):
    def __init__(self, dft_size=2000, max_sparsity=1000):
        super(ComplexDFTAttentionModel, self).__init__()
        
        # For 2000-point complex DFT (represented as 4000 real values)
        self.input_size = dft_size * 2
        self.output_size = dft_size * 2
        self.dft_size = dft_size
        
        # Encoder - processes real and imaginary parts together
        self.encoder = nn.Sequential(
            nn.Linear(self.input_size, 2048),
            nn.BatchNorm1d(2048),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3),
            
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3),
            
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.1),
        )
        
        # Self-attention for frequency domain relationships
        self.attention = FrequencyAttention(512, 8)
        
        # Decoder with frequency-aware processing
        self.decoder = nn.Sequential(
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3),
            
            nn.Linear(1024, 2048),
            nn.BatchNorm1d(2048),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3),
            
            nn.Linear(2048, self.output_size),
        )
        
        # Sparsity prediction takes the encoded representation
        self.sparsity_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3),
            nn.Linear(256, max_sparsity + 1)
        )
        
    def forward(self, x):
        # Encode the input
        features = self.encoder(x)
        
        # Apply attention mechanism
        attn_features = self.attention(features)
        
        # Decode to get complex DFT outputs
        dft_output = self.decoder(attn_features)
        
        # Predict sparsity
        sparsity_logits = self.sparsity_head(attn_features)
        
        return dft_output, sparsity_logits
    
    def complex_representation(self, x):
        """Helper method to convert the real 4000×1 vector back to 2000 complex values"""
        dft_output, _ = self.forward(x)
        batch_size = dft_output.shape[0]
        complex_output = torch.complex(
            dft_output[:, :self.dft_size], 
            dft_output[:, self.dft_size:]
        )
        return complex_output


class FrequencyAttention(nn.Module):
    """Custom attention mechanism for frequency domain data"""
    def __init__(self, embed_dim, num_heads):
        super(FrequencyAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        
        # Multi-head attention layer
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        
        # Projection layers to create query, key, value
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        
        # Output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, x):
        # Reshape for attention: [batch_size, 1, embed_dim]
        x_reshaped = x.unsqueeze(1)
        
        # Project to get query, key, value
        q = self.q_proj(x_reshaped)
        k = self.k_proj(x_reshaped)
        v = self.v_proj(x_reshaped)
        
        # Apply attention
        attn_output, _ = self.attention(q, k, v)
        
        # Reshape back and project
        attn_output = attn_output.squeeze(1)
        output = self.out_proj(attn_output)
        
        # Residual connection
        return x + output


class ComplexDFTDataset(Dataset):
    def __init__(self, inputs, targets_dft, targets_sparsity, max_sparsity=1000):
        """
        Args:
            inputs: Input DFT vectors [N, 4000] (real and imaginary parts)
            targets_dft: Target DFT vectors [N, 4000] (real and imaginary parts)
            targets_sparsity: Target sparsity integers [N]
            max_sparsity: Maximum sparsity value (for validation)
        """
        self.inputs = torch.FloatTensor(inputs)
        self.targets_dft = torch.FloatTensor(targets_dft)
        
        # Ensure sparsity targets are within range
        clipped_sparsity = np.clip(targets_sparsity, 0, max_sparsity)
        self.targets_sparsity = torch.LongTensor(clipped_sparsity)
    
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        return self.inputs[idx], self.targets_dft[idx], self.targets_sparsity[idx]


def train_complex_dft_model(model, train_loader, val_loader=None, epochs=50, lr=0.001, 
                           phase_weight=0.5, amplitude_weight=0.5):
    """
    Training function with special handling for complex DFT data
    
    Args:
        model: The model to train
        train_loader: DataLoader for training data
        val_loader: Optional DataLoader for validation
        epochs: Number of training epochs
        lr: Learning rate
        phase_weight: Weight for phase error in loss function
        amplitude_weight: Weight for amplitude error in loss function
    """
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        print(f"Training on {device}")
        
        # Loss functions
        # MSE for overall reconstruction
        mse_criterion = nn.MSELoss()
        # CrossEntropy for sparsity prediction
        sparsity_criterion = nn.CrossEntropyLoss()
        
        # Optimizer with weight decay for regularization
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
        
        # Learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
        
        # Store loss values
        train_losses = []
        train_dft_losses = []
        train_sparsity_losses = []
        val_losses = []


        # Training loop
        for epoch in range(epochs):
            model.train()
            running_loss = 0.0
            dft_loss_sum = 0.0
            sparsity_loss_sum = 0.0
            
            for inputs, targets_dft, targets_sparsity in train_loader:
                inputs = inputs.to(device)
                targets_dft = targets_dft.to(device)
                targets_sparsity = targets_sparsity.to(device)
                
                # Check if any target sparsity values are out of range
                max_target = targets_sparsity.max().item()
                min_target = targets_sparsity.min().item()
                output_size = model.sparsity_head[-1].out_features
                
                if max_target >= output_size or min_target < 0:
                    print(f"Warning: Target sparsity values out of range! Min: {min_target}, Max: {max_target}, Output size: {output_size}")
                    # Clip values to prevent errors
                    targets_sparsity = torch.clamp(targets_sparsity, 0, output_size - 1)
                
                # Zero the gradients
                optimizer.zero_grad()
                
                # Forward pass
                pred_dft, pred_sparsity = model(inputs)
                
                # DFT loss - standard MSE
                dft_loss = mse_criterion(pred_dft, targets_dft)
                
                # Calculate complex-aware losses (optional)
                dft_size = inputs.shape[1] // 2
                
                # Complex parts for inputs
                pred_real = pred_dft[:, :dft_size]
                pred_imag = pred_dft[:, dft_size:]
                target_real = targets_dft[:, :dft_size]
                target_imag = targets_dft[:, dft_size:]
                
                # Convert to complex numbers
                pred_complex = torch.complex(pred_real, pred_imag)
                target_complex = torch.complex(target_real, target_imag)
                
                # Amplitude loss
                pred_amplitude = torch.abs(pred_complex)
                target_amplitude = torch.abs(target_complex)
                amplitude_loss = mse_criterion(pred_amplitude, target_amplitude)
                
                # Phase loss (handling potential numerical issues)
                pred_phase = torch.angle(pred_complex)
                target_phase = torch.angle(target_complex)
                phase_loss = mse_criterion(pred_phase, target_phase)
                
                # Combined DFT loss with phase and amplitude components
                dft_combined_loss = dft_loss + amplitude_weight * amplitude_loss + phase_weight * phase_loss
                
                # Sparsity classification loss
                sparsity_loss = sparsity_criterion(pred_sparsity, targets_sparsity)
                
                # Total loss
                loss = dft_combined_loss + sparsity_loss
                
                # Backward pass and optimize
                loss.backward()
                # Gradient clipping to prevent explosion
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                # Track statistics
                running_loss += loss.item()
                dft_loss_sum += dft_combined_loss.item()
                sparsity_loss_sum += sparsity_loss.item()
            
            # Calculate average losses
            avg_loss = running_loss / len(train_loader)
            avg_dft_loss = dft_loss_sum / len(train_loader)
            avg_sparsity_loss = sparsity_loss_sum / len(train_loader)

            # Store values for plotting
            train_losses.append(avg_loss)
            train_dft_losses.append(avg_dft_loss)
            train_sparsity_losses.append(avg_sparsity_loss)
            
            # Update learning rate based on validation loss if provided
            if val_loader:
                val_loss = validate_complex_dft_model(model, val_loader, mse_criterion, 
                                                     sparsity_criterion, device,
                                                     phase_weight, amplitude_weight)
                scheduler.step(val_loss)
            else:
                scheduler.step(avg_loss)
            
            # Print epoch statistics
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, "
                  f"DFT Loss: {avg_dft_loss:.4f}, Sparsity Loss: {avg_sparsity_loss:.4f}")
            
        epochs_range = range(1, epochs + 1)

        plt.figure(figsize=(15, 5))

        # Total Loss Plot
        plt.subplot(1, 3, 1)
        plt.plot(epochs_range, train_losses, label="Total Loss", color='blue')
        if val_loader:
            plt.plot(epochs_range, val_losses, label="Validation Loss", color='red', linestyle="--")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Total Loss")
        plt.legend()
        plt.grid()

        # DFT Loss Plot
        plt.subplot(1, 3, 2)
        plt.plot(epochs_range, train_dft_losses, label="DFT Loss", color='orange')
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("DFT Loss")
        plt.legend()
        plt.grid()

        # Sparsity Loss Plot
        plt.subplot(1, 3, 3)
        plt.plot(epochs_range, train_sparsity_losses, label="Sparsity Loss", color='green')
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Sparsity Loss")
        plt.legend()
        plt.grid()

        # Show all plots
        plt.tight_layout()
        plt.show()
        
        return model
    
    except Exception as e:
        print(f"Error during training: {str(e)}")
        import traceback
        traceback.print_exc()
        return model


def validate_complex_dft_model(model, val_loader, mse_criterion, sparsity_criterion, 
                              device, phase_weight=0.5, amplitude_weight=0.5):
    """Validation function for complex DFT model"""
    model.eval()
    val_loss = 0.0
    dft_loss_sum = 0.0
    sparsity_loss_sum = 0.0
    sparsity_correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets_dft, targets_sparsity in val_loader:
            inputs = inputs.to(device)
            targets_dft = targets_dft.to(device)
            targets_sparsity = targets_sparsity.to(device)
            
            # Clip values to prevent errors
            output_size = model.sparsity_head[-1].out_features
            targets_sparsity = torch.clamp(targets_sparsity, 0, output_size - 1)
            
            # Forward pass
            pred_dft, pred_sparsity = model(inputs)
            
            # DFT loss - standard MSE
            dft_loss = mse_criterion(pred_dft, targets_dft)
            
            # Calculate complex-aware losses
            dft_size = inputs.shape[1] // 2
            
            # Complex parts
            pred_real = pred_dft[:, :dft_size]
            pred_imag = pred_dft[:, dft_size:]
            target_real = targets_dft[:, :dft_size]
            target_imag = targets_dft[:, dft_size:]
            
            # Convert to complex
            pred_complex = torch.complex(pred_real, pred_imag)
            target_complex = torch.complex(target_real, target_imag)
            
            # Amplitude loss
            pred_amplitude = torch.abs(pred_complex)
            target_amplitude = torch.abs(target_complex)
            amplitude_loss = mse_criterion(pred_amplitude, target_amplitude)
            
            # Phase loss
            pred_phase = torch.angle(pred_complex)
            target_phase = torch.angle(target_complex)
            phase_loss = mse_criterion(pred_phase, target_phase)
            
            # Combined DFT loss
            dft_combined_loss = dft_loss + amplitude_weight * amplitude_loss + phase_weight * phase_loss
            
            # Sparsity loss
            sparsity_loss = sparsity_criterion(pred_sparsity, targets_sparsity)
            
            # Total loss
            loss = dft_combined_loss + sparsity_loss
            
            # Track statistics
            val_loss += loss.item()
            dft_loss_sum += dft_combined_loss.item()
            sparsity_loss_sum += sparsity_loss.item()
            
            # Calculate accuracy for sparsity prediction
            _, predicted = torch.max(pred_sparsity, 1)
            total += targets_sparsity.size(0)
            sparsity_correct += (predicted == targets_sparsity).sum().item()
    
    # Calculate average losses
    avg_val_loss = val_loss / len(val_loader)
    avg_dft_loss = dft_loss_sum / len(val_loader)
    avg_sparsity_loss = sparsity_loss_sum / len(val_loader)
    accuracy = 100 * sparsity_correct / total
    
    print(f"Validation - Loss: {avg_val_loss:.4f}, "
          f"DFT Loss: {avg_dft_loss:.4f}, Sparsity Loss: {avg_sparsity_loss:.4f}, "
          f"Sparsity Accuracy: {accuracy:.2f}%")
    
    return avg_val_loss

# Code to save the trained model
def save_model(model, path="complex_dft_model.pth"):
    """
    Save the trained model to a file
    
    Args:
        model: The trained model
        path: Path where to save the model
    """
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")


# Code to load the saved model
def load_model(model_class, path="complex_dft_model.pth", dft_size=2000, max_sparsity=1000):
    """
    Load a saved model
    
    Args:
        model_class: The model class (ComplexDFTModel or ComplexDFTAttentionModel)
        path: Path to the saved model file
        dft_size: DFT size parameter for model initialization
        max_sparsity: Max sparsity parameter for model initialization
        
    Returns:
        Loaded model
    """
    model = model_class(dft_size=dft_size, max_sparsity=max_sparsity)
    model.load_state_dict(torch.load(path))
    model.eval()  # Set to evaluation mode
    return model


# Function to use the model for predictions
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
        dft_output, sparsity_logits = model(input_tensor)
        
        # Get predicted sparsity class
        _, predicted_sparsity = torch.max(sparsity_logits, 1)
    
    # Return single sample or batch depending on input
    if is_single_sample:
        return dft_output.squeeze(0).cpu().numpy(), predicted_sparsity.item()
    else:
        return dft_output.cpu().numpy(), predicted_sparsity.cpu().numpy()

