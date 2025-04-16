import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
from torch.utils.data import Dataset, DataLoader, TensorDataset
from tqdm import tqdm
import soundfile as sf
from time import time
from scipy.signal import resample
from spade_segmentation import spade_segmentation_eval
import pandas as pd
from clip_sdr_modified import clip_sdr_modified
from typing import Tuple, List, Optional, Dict
from sdr import sdr
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional, Dict
from ML_model import ASPADEDataset,ASPADEModel
import sys
import librosa
from pesq import pesq



def preprocess_data(training_data):
    inputs = []
    targets = []
    
    for data_point in training_data:
        initial_estimate = data_point[0]  # 8000xn complex initial estimate
        best_estimate = data_point[1]     # 8000xn complex best estimate
        best_sparsity = data_point[2]     # 1xn best sparsity
        
        # Convert complex arrays to real representation
        # For each complex number, we'll have its real and imaginary parts
        inputs.append(np.hstack([initial_estimate.real, initial_estimate.imag]))
        targets.append((np.hstack([best_estimate.real, best_estimate.imag]), best_sparsity))
    
    inputs = np.array(inputs)
    targets_estimates = np.array([t[0] for t in targets])
    targets_sparsity = np.array([t[1] for t in targets], dtype=np.int64)  # Ensure this is integer
    
    return inputs, targets_estimates, targets_sparsity


def train_and_save_model(training_data, model_path, num_epochs=50, lr=0.0005, batch_size=16):
    sample_initial = training_data[0][0]  
    input_dim = 2 * sample_initial.shape[0]  # Multiply by 2 for real and imaginary parts
    output_dim = input_dim  
    max_sparsity = int(input_dim//4)
    
    print(f"Input dimension: {input_dim}, Output dimension: {output_dim}")
    
    # Initialize model with GPU support if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = ASPADEModel(input_dim, output_dim, max_sparsity).to(device)
    
    # Use gradient clipping to prevent exploding gradients
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
    
    estimate_criterion = nn.MSELoss()
    sparsity_criterion = nn.CrossEntropyLoss()  
    
    inputs, targets_estimates, targets_sparsity = preprocess_data(training_data)



    dataset = ASPADEDataset(inputs, targets_estimates, targets_sparsity)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4 if torch.cuda.is_available() else 0)
    
    epoch_losses = []
    epoch_estimate_losses = []
    epoch_sparsity_losses = []
    epoch_accuracies = []
    
    best_loss = float('inf')
    patience_counter = 0
    patience = 50  # Early stopping patience
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        running_estimate_loss = 0.0
        running_sparsity_loss = 0.0
        total_correct = 0
        total_samples = 0
        batch_count = 0
        
        for batch_x, batch_y, batch_k in dataloader:
            batch_x, batch_y, batch_k = batch_x.to(device), batch_y.to(device), batch_k.to(device)
            
            optimizer.zero_grad()
            pred_output, pred_k_logits = model(batch_x)
            
            loss_estimate = estimate_criterion(pred_output, batch_y)
            loss_sparsity = sparsity_criterion(pred_k_logits, batch_k)
            
            # Balance between the two losses
            loss = loss_estimate + 0.1 * loss_sparsity
            
            loss.backward()
            # Add gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            _, predicted_k = torch.max(pred_k_logits, 1)
            total_correct += (predicted_k == batch_k).sum().item()
            total_samples += batch_k.size(0)
            
            running_loss += loss.item()
            running_estimate_loss += loss_estimate.item()
            running_sparsity_loss += loss_sparsity.item()
            batch_count += 1
            
            # Overwriting progress update
            sparsity_accuracy = 100 * total_correct / total_samples
            sys.stdout.write(
                f"\rEpoch {epoch+1}/{num_epochs} | "
                f"Loss: {running_loss/batch_count:.4f} | "
                f"Est: {running_estimate_loss/batch_count:.4f} | "
                f"Spar: {running_sparsity_loss/batch_count:.4f} | "
                f"Acc: {sparsity_accuracy:.2f}%"
            )
            sys.stdout.flush()
        
        avg_loss = running_loss / batch_count
        epoch_losses.append(avg_loss)
        epoch_estimate_losses.append(running_estimate_loss / batch_count)
        epoch_sparsity_losses.append(running_sparsity_loss / batch_count)
        epoch_accuracies.append(sparsity_accuracy)
        
        # Update learning rate based on loss
        scheduler.step(avg_loss)
        
        # Save the best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), model_path)
            patience_counter = 0
            print(f"\nEpoch {epoch+1}: New best model saved with loss {best_loss:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch+1} as no improvement for {patience} epochs")
                break
            
    print("\nTraining complete.")
    
    # Load the best model for evaluation
    model.load_state_dict(torch.load(model_path))
    
    # Plot loss and accuracy curves
    fig, axes = plt.subplots(4, 1, figsize=(10, 16), sharex=True)
    
    axes[0].plot(range(1, len(epoch_losses)+1), epoch_losses, color='b')
    axes[0].set_title('Total Loss')
    axes[0].set_ylabel('Loss')
    axes[0].grid(True)

    axes[1].plot(range(1, len(epoch_estimate_losses)+1), epoch_estimate_losses, color='g')
    axes[1].set_title('Estimate Loss')
    axes[1].set_ylabel('Loss')
    axes[1].grid(True)

    axes[2].plot(range(1, len(epoch_sparsity_losses)+1), epoch_sparsity_losses, color='r')
    axes[2].set_title('Sparsity Loss')
    axes[2].set_ylabel('Loss')
    axes[2].grid(True)

    axes[3].plot(range(1, len(epoch_accuracies)+1), epoch_accuracies, color='m')
    axes[3].set_title('Sparsity Prediction Accuracy')
    axes[3].set_xlabel('Epoch')
    axes[3].set_ylabel('Accuracy (%)')
    axes[3].grid(True)

    plt.tight_layout()
    plt.savefig('models/training_curves.png')
    plt.show()
    
    return model




def evaluate_model(test_audio_dir: str,
                   output_dir: str,
                   target_fs_values: List[int],
                   clipping_thresholds: List[float],
                   time_clip: List[int],
                   model_path: str,
                   loaded_model, 
                   device) -> Dict:

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Initialize results storage
    results = {
        'file': [],
        'fs': [],
        'threshold': [],
        'duration': [],
        'sdr_original': [],
        'sdr_reconstructed': [],
        'sdr_improvement': [],
        'processing_time': [],
        'clipped_percentage': [],
        'pesq_i': [],
        'pesq_f': []
    }

    # Get all WAV files
    wav_files = [f for f in os.listdir(test_audio_dir) if f.endswith(".wav")]
    total_configs = len(target_fs_values) * len(clipping_thresholds) * len(time_clip) * len(wav_files)
    pbar = tqdm(total=total_configs, desc="Processing files")

    for target_fs in target_fs_values:
        for clipping_threshold in clipping_thresholds:
            for tc in time_clip:
                for audio_file in wav_files:
                    print(f"\nProcessing: {audio_file} (fs={target_fs}, threshold={clipping_threshold}, duration={tc})")

                    audio_path = os.path.join(test_audio_dir, audio_file)

                    try:
                        # Load and preprocess audio
                        data, fs = sf.read(audio_path)
                    except Exception as e:
                        print(f"Error reading {audio_file}: {e}")
                        pbar.update(1)
                        continue
                    
                    if len(data.shape) > 1:
                        data = data[:, 0]  # Convert stereo to mono

                    # Clip to desired duration and normalize
                    max_samples = min(len(data), fs * tc)
                    data = data[:max_samples]
                    if np.max(np.abs(data)) > 0:  # Prevent division by zero
                        data = data / np.max(np.abs(data))

                    # Resample to target frequency
                    resampled_data = resample(data, int(target_fs * tc))

                    # Setup parameters
                    Ls = len(resampled_data)
                    win_len = np.floor(Ls / 32)
                    win_shift = np.floor(win_len / 4)
                    F_red = 2

                    # ASPADE parameters
                    ps_s = 1
                    ps_r = 2
                    ps_epsilon = 0.1
                    ps_maxit = np.ceil(np.floor(win_len * F_red / 2 + 1) * ps_r / ps_s)

                    # Generate clipped signal
                    print("Generating clipped signal...")
                    clipped_signal, masks, theta, sdr_original, clipped_percentage = \
                        clip_sdr_modified(resampled_data, clipping_threshold)

                    # Perform reconstruction
                    start_time = time()
                    reconstructed_signal = spade_segmentation_eval(
                        clipped_signal, resampled_data, Ls, win_len, win_shift,
                        ps_maxit, ps_epsilon, ps_r, ps_s, F_red, masks, model_path,
                        loaded_model, device
                    )
                    processing_time = time() - start_time

                    # Resample back to target_fs (not original fs)
                    reconstructed_signal = resample(reconstructed_signal, int(fs * tc))

                    clipped_signal = resample(clipped_signal, int(fs * tc))




                    pesq_i = pesq(16000, data, clipped_signal, 'wb')
                    pesq_f = pesq(16000, data, reconstructed_signal, 'wb')

                    # Calculate metrics
                    sdr_reconstructed = sdr(data, reconstructed_signal)
                    sdr_improvement = sdr_reconstructed - sdr_original

                    # Save reconstructed audio
                    dir_name = f"fs_{fs}_threshold_{clipping_threshold:.2f}"
                    full_dir_path = os.path.join(output_dir, dir_name)
                    os.makedirs(full_dir_path, exist_ok=True)
                    output_path = os.path.join(full_dir_path, f"reconstructed_{audio_file}")
                    sf.write(output_path, reconstructed_signal, fs)

                    # Store results
                    results['file'].append(audio_file)
                    results['fs'].append(fs)
                    results['threshold'].append(clipping_threshold)
                    results['duration'].append(tc)
                    results['sdr_original'].append(sdr_original)
                    results['sdr_reconstructed'].append(sdr_reconstructed)
                    results['sdr_improvement'].append(sdr_improvement)
                    results['processing_time'].append(processing_time)
                    results['clipped_percentage'].append(clipped_percentage)
                    results['pesq_i'].append(pesq_i)
                    results['pesq_f'].append(pesq_f)

                    # # Plot the reconstructed signal
                    # plt.figure(figsize=(8, 4))
                    # plt.plot(reconstructed_signal, color='green', linewidth=1.5)
                    # plt.title("Reconstructed Signal (ML)")
                    # plt.xlabel("Time")
                    # plt.ylabel("Amplitude")
                    # plt.ylim(-1, 1)  # Set y-axis limits with a margin
                    # plt.grid(True, linestyle='--', alpha=0.6)
                    # plt.show()

                    pbar.update(1)  # Update progress bar after each file

    pbar.close()

    # Convert results to DataFrame and save
    results_df = pd.DataFrame(results)
    
    # Print final results to check
    print("\nFinal Results DataFrame:")
    print(results_df)

    # Save CSV for verification
    results_csv_path = os.path.join(output_dir, 'evaluation_results.csv')
    results_df.to_csv(results_csv_path, index=False)

    # Generate summary statistics
    summary = results_df.groupby(['fs', 'threshold', 'duration']).agg({
        'sdr_improvement': ['mean', 'std'],
        'processing_time': 'mean',
        'clipped_percentage': 'mean',
        'pesq_i': 'mean',
        'pesq_f': 'mean'
    }).round(2)

    print("\nSummary Statistics:")
    print(summary)

    return results_df, summary
