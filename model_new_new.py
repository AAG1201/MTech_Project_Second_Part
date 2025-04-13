import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset


class ComplexDFTUNet(nn.Module):
    def __init__(self, dft_size=2000, max_sparsity=1000):
        super(ComplexDFTUNet, self).__init__()
        
        # For 2000-point complex DFT (represented as 4000 real values)
        self.input_size = dft_size * 2
        self.output_size = dft_size * 2
        self.dft_size = dft_size
        
        # Encoder path (downsampling)
        self.enc1 = nn.Sequential(
            nn.Linear(self.input_size, 2048),
            nn.BatchNorm1d(2048),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2)
        )
        
        self.enc2 = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2)
        )
        
        self.enc3 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2)
        )
        
        self.enc4 = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.1)
        )
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.1)
        )
        
        # Decoder path (upsampling) with skip connections
        self.dec4 = nn.Sequential(
            nn.Linear(256 + 256, 512),  # Skip connection from enc3
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2)
        )
        
        self.dec3 = nn.Sequential(
            nn.Linear(512 + 512, 1024),  # Skip connection from enc2
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2)
        )
        
        self.dec2 = nn.Sequential(
            nn.Linear(1024 + 1024, 2048),  # Skip connection from enc1
            nn.BatchNorm1d(2048),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2)
        )
        
        self.dec1 = nn.Sequential(
            nn.Linear(2048 + 2048, 2048),  # Skip connection from input
            nn.BatchNorm1d(2048),
            nn.LeakyReLU(0.1),
            nn.Linear(2048, self.output_size)
        )
        
        # Attention modules for skip connections
        self.attention1 = FrequencyAttentionGate(2048, 2048)
        self.attention2 = FrequencyAttentionGate(1024, 1024)
        self.attention3 = FrequencyAttentionGate(512, 512)
        
        # Sparsity prediction branch from bottleneck features
        self.sparsity_head = nn.Sequential(
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),
            nn.Linear(256, max_sparsity + 1)
        )
    
    def forward(self, x):
        # Encoder path
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        
        # Bottleneck
        b = self.bottleneck(e4)
        
        # Decoder path with skip connections and attention gates
        # Apply attention to skip connections
        a3 = self.attention3(e3, b)
        d4 = self.dec4(torch.cat([b, a3], dim=1))
        
        a2 = self.attention2(e2, d4)
        d3 = self.dec3(torch.cat([d4, a2], dim=1))
        
        a1 = self.attention1(e1, d3)
        d2 = self.dec2(torch.cat([d3, a1], dim=1))
        
        # Final decoder layer with direct skip connection
        d1 = self.dec1(torch.cat([d2, e1], dim=1))
        
        # Sparsity prediction from bottleneck features
        sparsity_logits = self.sparsity_head(e4)
        
        return d1, sparsity_logits
    
    def complex_representation(self, x):
        """Helper method to convert the real vector back to complex values"""
        dft_output, _ = self.forward(x)
        batch_size = dft_output.shape[0]
        complex_output = torch.complex(
            dft_output[:, :self.dft_size], 
            dft_output[:, self.dft_size:]
        )
        return complex_output


class FrequencyAttentionGate(nn.Module):
    """Attention Gate for U-Net skip connections"""
    def __init__(self, F_g, F_l):
        super(FrequencyAttentionGate, self).__init__()
        self.W_g = nn.Sequential(
            nn.Linear(F_g, F_l),
            nn.BatchNorm1d(F_l)
        )
        
        self.W_x = nn.Sequential(
            nn.Linear(F_l, F_l),
            nn.BatchNorm1d(F_l)
        )
        
        self.psi = nn.Sequential(
            nn.Linear(F_l, 1),
            nn.BatchNorm1d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU()
        
    def forward(self, g, x):
        """
        g: gating signal from coarser level (decoder features)
        x: skip connection features from encoder
        """
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        
        return x * psi


class CustomFrequencyLoss(nn.Module):
    """Custom loss function for DFT reconstruction with frequency-dependent weighting"""
    def __init__(self, dft_size=2000, alpha=0.7, beta=1.0, gamma=1.3):
        super(CustomFrequencyLoss, self).__init__()
        self.dft_size = dft_size
        self.alpha = alpha  # Weight for MSE loss
        self.beta = beta    # Weight for magnitude loss
        self.gamma = gamma  # Weight for phase loss
        
        # Create frequency-dependent weights (emphasize lower frequencies)
        freq_weights = torch.ones(dft_size)
        for i in range(dft_size):
            # Higher weight for lower frequencies (can be customized)
            freq_weights[i] = 1.0 + 0.5 * (1.0 - i / dft_size)
        self.register_buffer('freq_weights', freq_weights)
    
    def forward(self, output, target):
        # Split real and imaginary parts
        real_output = output[:, :self.dft_size]
        imag_output = output[:, self.dft_size:]
        real_target = target[:, :self.dft_size]
        imag_target = target[:, self.dft_size:]
        
        # Convert to complex representation
        complex_output = torch.complex(real_output, imag_output)
        complex_target = torch.complex(real_target, imag_target)
        
        # MSE loss on raw real/imag values
        mse_loss = F.mse_loss(output, target)
        
        # Magnitude loss (weighted by frequency importance)
        output_mag = torch.abs(complex_output)
        target_mag = torch.abs(complex_target)
        mag_diff = (output_mag - target_mag).pow(2)
        weighted_mag_loss = (mag_diff * self.freq_weights.unsqueeze(0)).mean()
        
        # Phase loss (only where magnitude is significant)
        # Avoid phase issues where magnitude is near zero
        significant_idx = target_mag > 0.05 * torch.max(target_mag, dim=1, keepdim=True)[0]
        if significant_idx.sum() > 0:
            output_phase = torch.angle(complex_output)
            target_phase = torch.angle(complex_target)
            
            # Handle phase wrapping
            phase_diff = torch.abs(output_phase - target_phase)
            phase_diff = torch.min(phase_diff, 2*torch.pi - phase_diff)
            phase_loss = (phase_diff[significant_idx]).pow(2).mean()
        else:
            phase_loss = torch.tensor(0.0, device=output.device)
        
        # Combined loss
        total_loss = self.alpha * mse_loss + self.beta * weighted_mag_loss + self.gamma * phase_loss
        
        return total_loss


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


def train_complex_dft_unet(model, train_loader, val_loader=None, epochs=50, lr=0.001, 
                           use_custom_loss=True, phase_weight=0.5, amplitude_weight=0.5,
                           frequency_alpha=0.7, frequency_beta=1.0, frequency_gamma=1.3):
    """
    Training function for ComplexDFTUNet with custom frequency loss option
    
    Args:
        model: The model to train (ComplexDFTUNet)
        train_loader: DataLoader for training data
        val_loader: Optional DataLoader for validation
        epochs: Number of training epochs
        lr: Learning rate
        use_custom_loss: Whether to use CustomFrequencyLoss
        phase_weight: Weight for phase error in standard loss function
        amplitude_weight: Weight for amplitude error in standard loss function
        frequency_alpha: Alpha parameter for CustomFrequencyLoss
        frequency_beta: Beta parameter for CustomFrequencyLoss
        frequency_gamma: Gamma parameter for CustomFrequencyLoss
    """
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        print(f"Training on {device}")
        dft_size = model.dft_size
        
        # Choose loss function
        if use_custom_loss:
            print("Using CustomFrequencyLoss")
            dft_criterion = CustomFrequencyLoss(dft_size=dft_size, 
                                              alpha=frequency_alpha, 
                                              beta=frequency_beta, 
                                              gamma=frequency_gamma).to(device)
            # For backward compatibility during validation
            mse_criterion = nn.MSELoss()
        else:
            print("Using standard MSE with phase and amplitude components")
            dft_criterion = nn.MSELoss()
            mse_criterion = dft_criterion
        
        # CrossEntropy for sparsity prediction
        sparsity_criterion = nn.CrossEntropyLoss()
        
        # Optimizer with weight decay for regularization
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
        
        # Learning rate scheduler with warmup and cosine annealing
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
        
        # Store loss values
        train_losses = []
        train_dft_losses = []
        train_sparsity_losses = []
        val_losses = []
        val_dft_losses = []
        val_sparsity_losses = []

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
                
                if use_custom_loss:
                    # Use the custom frequency loss for DFT
                    dft_loss = dft_criterion(pred_dft, targets_dft)
                else:
                    # Standard approach with component-wise losses
                    # DFT loss - standard MSE
                    dft_loss = mse_criterion(pred_dft, targets_dft)
                    
                    # Calculate complex-aware losses (for standard approach)
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
                    dft_loss = dft_loss + amplitude_weight * amplitude_loss + phase_weight * phase_loss
                
                # Sparsity classification loss
                sparsity_loss = sparsity_criterion(pred_sparsity, targets_sparsity)
                
                # Total loss
                loss = dft_loss + sparsity_loss
                
                # Backward pass and optimize
                loss.backward()
                # Gradient clipping to prevent explosion
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                # Track statistics
                running_loss += loss.item()
                dft_loss_sum += dft_loss.item()
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
                val_loss, val_dft, val_sparsity = validate_complex_dft_unet(
                    model, val_loader, use_custom_loss, dft_criterion, mse_criterion,
                    sparsity_criterion, device, phase_weight, amplitude_weight
                )
                val_losses.append(val_loss)
                val_dft_losses.append(val_dft)
                val_sparsity_losses.append(val_sparsity)
                scheduler.step(val_loss)
            else:
                scheduler.step(avg_loss)
            
            # Print epoch statistics
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, "
                  f"DFT Loss: {avg_dft_loss:.4f}, Sparsity Loss: {avg_sparsity_loss:.4f}")
            
        # Plot training statistics
        plot_training_stats(
            epochs, train_losses, train_dft_losses, train_sparsity_losses,
            val_losses if val_loader else None,
            val_dft_losses if val_loader else None,
            val_sparsity_losses if val_loader else None
        )
        
        return model
    
    except Exception as e:
        print(f"Error during training: {str(e)}")
        import traceback
        traceback.print_exc()
        return model


def validate_complex_dft_unet(model, val_loader, use_custom_loss, dft_criterion, 
                            mse_criterion, sparsity_criterion, device, 
                            phase_weight=0.5, amplitude_weight=0.5):
    """Validation function for complex DFT U-Net model"""
    model.eval()
    val_loss = 0.0
    dft_loss_sum = 0.0
    sparsity_loss_sum = 0.0
    sparsity_correct = 0
    total = 0
    dft_size = model.dft_size
    
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
            
            if use_custom_loss:
                # Use the custom frequency loss for DFT
                dft_loss = dft_criterion(pred_dft, targets_dft)
            else:
                # Standard approach with component-wise losses
                # DFT loss - standard MSE
                dft_loss = mse_criterion(pred_dft, targets_dft)
                
                # Calculate complex-aware losses
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
                dft_loss = dft_loss + amplitude_weight * amplitude_loss + phase_weight * phase_loss
            
            # Sparsity loss
            sparsity_loss = sparsity_criterion(pred_sparsity, targets_sparsity)
            
            # Total loss
            loss = dft_loss + sparsity_loss
            
            # Track statistics
            val_loss += loss.item()
            dft_loss_sum += dft_loss.item()
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
    
    return avg_val_loss, avg_dft_loss, avg_sparsity_loss


def plot_training_stats(epochs, train_losses, train_dft_losses, train_sparsity_losses,
                       val_losses=None, val_dft_losses=None, val_sparsity_losses=None):
    """Plot training statistics for visualization"""
    epochs_range = range(1, epochs + 1)

    plt.figure(figsize=(15, 5))

    # Total Loss Plot
    plt.subplot(1, 3, 1)
    plt.plot(epochs_range, train_losses, label="Train Loss", color='blue')
    if val_losses:
        plt.plot(epochs_range, val_losses, label="Validation Loss", color='red', linestyle="--")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Total Loss")
    plt.legend()
    plt.grid()

    # DFT Loss Plot
    plt.subplot(1, 3, 2)
    plt.plot(epochs_range, train_dft_losses, label="Train DFT Loss", color='orange')
    if val_dft_losses:
        plt.plot(epochs_range, val_dft_losses, label="Val DFT Loss", color='darkred', linestyle="--")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("DFT Loss")
    plt.legend()
    plt.grid()

    # Sparsity Loss Plot
    plt.subplot(1, 3, 3)
    plt.plot(epochs_range, train_sparsity_losses, label="Train Sparsity", color='green')
    if val_sparsity_losses:
        plt.plot(epochs_range, val_sparsity_losses, label="Val Sparsity", color='darkgreen', linestyle="--")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Sparsity Loss")
    plt.legend()
    plt.grid()

    # Show all plots
    plt.tight_layout()
    plt.show()


# Code to save the trained model
def save_model(model, path="complex_dft_unet_model.pth"):
    """
    Save the trained model to a file
    
    Args:
        model: The trained model
        path: Path where to save the model
    """
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")


# Code to load the saved model
def load_model(model_class, path="complex_dft_unet_model.pth", dft_size=2000, max_sparsity=1000):
    """
    Load a saved model
    
    Args:
        model_class: The model class (ComplexDFTUNet)
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


# Example usage
def train_and_evaluate(train_dataset, val_dataset=None, batch_size=64, epochs=50):
    """
    Example function to train and evaluate the U-Net model
    
    Args:
        train_dataset: Training dataset (ComplexDFTDataset)
        val_dataset: Validation dataset (optional)
        batch_size: Batch size for training
        epochs: Number of training epochs
    """
    from torch.utils.data import DataLoader
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size) if val_dataset else None
    
    # Create model
    dft_size = train_dataset.inputs.shape[1] // 2
    max_sparsity = train_dataset.targets_sparsity.max().item()
    
    model = ComplexDFTUNet(dft_size=dft_size, max_sparsity=max_sparsity+1)
    
    # Train model with custom frequency loss
    model = train_complex_dft_unet(
        model, 
        train_loader, 
        val_loader, 
        epochs=epochs, 
        lr=0.001,
        use_custom_loss=True,  # Use the specialized loss function
        phase_weight=0.7,
        amplitude_weight=1.2,
        frequency_alpha=0.7,
        frequency_beta=1.2,
        frequency_gamma=1.5
    )
    
    # Save model
    save_model(model, "complex_dft_unet_best.pth")
    
    return model