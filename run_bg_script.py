import pickle
import os
import shutil
import multiprocessing
from typing import List
from time import time
import numpy as np
from training_data_gen import training_data_func

def process_batch(batch_params):
    """Process a single batch of data with specific parameters"""
    audio_files, audio_dir, output_path, target_fs, clipping_threshold, time_clip, batch_id = batch_params
    
    print(f"Processing batch {batch_id}: fs={target_fs}, clip={clipping_threshold}, {len(audio_files)} files", flush=True)
    
    # Create a temporary directory for this batch
    batch_audio_dir = os.path.join(output_path, f"temp_batch_{batch_id}")
    os.makedirs(batch_audio_dir, exist_ok=True)
    
    # Copy files to temporary directory
    try:
        # Copy audio files to the temporary directory
        for audio_file in audio_files:
            # Get just the filename, not the path
            filename = os.path.basename(audio_file)
            dest_path = os.path.join(batch_audio_dir, filename)
            shutil.copy2(audio_file, dest_path)
        
        # Create a subfolder for each configuration
        config_folder = f"fs{target_fs}_clip{clipping_threshold:.2f}_batch{batch_id}"
        config_path = os.path.join(output_path, config_folder)
        os.makedirs(config_path, exist_ok=True)
        
        # Run training pipeline on the temporary directory
        training_data = training_data_func(
            audio_dir=batch_audio_dir,
            output_path=config_path,
            target_fs_values=[target_fs],
            clipping_thresholds=[clipping_threshold],
            time_clip=[time_clip]
        )
        
        # Save using pickle
        output_file = os.path.join(config_path, f'training_data_batch{batch_id}.pkl')
        with open(output_file, 'wb') as f:
            pickle.dump(training_data, f)
        
        print(f"Completed batch {batch_id}: fs={target_fs}, clip={clipping_threshold}", flush=True)
        return output_file
    
    finally:
        # Clean up - remove temporary directory
        shutil.rmtree(batch_audio_dir, ignore_errors=True)

def main():
    audio_dir = "train_data"
    output_path = "training_data"
    target_fs_values = [16000]
    clipping_thresholds = [0.1, 0.2, 0.3]
    time_clip = [1]
    
    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)
    
    # Get list of all audio files
    all_audio_files = []
    for root, _, files in os.walk(audio_dir):
        for file in files:
            if file.endswith(".wav"):  # Only looking for .wav files per the function
                all_audio_files.append(os.path.join(root, file))
    
    print(f"Found {len(all_audio_files)} audio files", flush=True)
    
    # Split files into 4 batches (25% each)
    total_files = len(all_audio_files)
    batch_size = total_files // 4
    file_batches = [
        all_audio_files[0:batch_size],
        all_audio_files[batch_size:2*batch_size],
        all_audio_files[2*batch_size:3*batch_size],
        all_audio_files[3*batch_size:]  # This last batch might have slightly more files if total_files isn't divisible by 4
    ]
    
    # Print batch sizes for verification
    for i, batch in enumerate(file_batches):
        print(f"Batch {i+1} size: {len(batch)} files", flush=True)
    
    # Prepare batch parameters
    batch_params = []
    batch_id = 1
    
    for target_fs in target_fs_values:
        for clipping_threshold in clipping_thresholds:
            for batch in file_batches:
                batch_params.append((
                    batch, 
                    audio_dir, 
                    output_path, 
                    target_fs, 
                    clipping_threshold, 
                    time_clip[0], 
                    batch_id
                ))
                batch_id += 1
    
    # Process batches in parallel with a limit of 6 concurrent processes
    with multiprocessing.Pool(processes=6) as pool:
        output_files = pool.map(process_batch, batch_params)
    
    print(f"All processing complete. Created files:", flush=True)
    for file in output_files:
        print(f"  - {file}", flush=True)
    
    # Optionally, combine all pickled results into one final file
    print("Combining results...", flush=True)
    combined_training_data = []
    for file in output_files:
        with open(file, 'rb') as f:
            batch_data = pickle.load(f)
            combined_training_data.extend(batch_data)
    
    # Save the combined data
    combined_output_file = os.path.join(output_path, 'combined_training_data.pkl')
    with open(combined_output_file, 'wb') as f:
        pickle.dump(combined_training_data, f)
    
    print(f"Combined data saved to {combined_output_file}", flush=True)

if __name__ == "__main__":
    main()