import numpy as np
import os
import soundfile as sf
from scipy.signal import resample
from time import time
from tqdm import tqdm
from clip_sdr_modified import clip_sdr_modified
from spade_segmentation import spade_segmentation_train
from typing import Tuple, List, Optional, Dict
from sdr import sdr
import pickle
import shutil
import random
import argparse
import multiprocessing


def training_data_func(audio_dir: str,
                      output_path: str,
                      target_fs_values: List[int],
                      clipping_thresholds: List[float],
                      time_clip: List[int],
                      win_len: int,
                      win_shift: int,
                      delta: int
                      ):
    """
    Complete training pipeline for ASPADE ML enhancement
    """
    
    training_data = []
    
    total_combinations = len(target_fs_values) * len(clipping_thresholds) * len(time_clip)
    
    for target_fs in target_fs_values:
        for clipping_threshold in clipping_thresholds:
            dir_name = f"fs_{target_fs}_threshold_{clipping_threshold:.2f}"
            full_dir_path = os.path.join(output_path, dir_name)
            os.makedirs(full_dir_path, exist_ok=True)

            for tc in time_clip:
                wav_files = [f for f in os.listdir(audio_dir) if f.endswith(".wav")]
                n_files = len(wav_files)
                wav_files = wav_files[:n_files]

                for i, audio_file in enumerate(wav_files):
                    print(f"\nProcessing: {audio_file}", flush=True)
                    data, fs = sf.read(os.path.join(audio_dir, audio_file))
                    
                    # Preprocessing steps
                    if len(data.shape) > 1:
                        data = data[:, 0]
                    
                    data = data[delta : ((fs * tc) + delta)]
                    
                    data = data / max(np.abs(data))  
                    
                    resampled_data = resample(data, int(target_fs * tc))
                    
                    # Setup parameters
                    Ls = len(resampled_data)
                    # win_len = np.floor(Ls/K)
                    # win_shift = np.floor(win_len/4)
                    F_red = 2
                    
                    # ASPADE parameters
                    ps_s = 1
                    ps_r = 2
                    ps_epsilon = 0.1
                    ps_maxit = np.ceil(np.floor(win_len * F_red / 2 + 1) * ps_r / ps_s)

                    # Generate clipped signal
                    clipped_signal, masks, theta, true_sdr, percentage = clip_sdr_modified(resampled_data, clipping_threshold)
                    
                    # Reconstruction and timing
                    start_time = time()
                    reconstructed_signal, metrics, intermediate_training_data, _ = spade_segmentation_train(
                        clipped_signal, resampled_data, Ls, win_len, win_shift,
                        ps_maxit, ps_epsilon, ps_r, ps_s, F_red, masks
                    )
                    elapsed_time = time() - start_time

                    reconstructed_signal = resample(reconstructed_signal, int(fs * tc))

                    # Calculate metrics
                    sdr_reconstructed = sdr(data, reconstructed_signal)
                    sdr_improvement = sdr_reconstructed - true_sdr
                    print(f"Done with SDR imp:{sdr_improvement:.2f} dB and {percentage:.2f}% and {true_sdr:.2f} dB", flush=True)

                    training_data.extend(intermediate_training_data)
    
    return training_data


def process_batch(batch_params):
    """Process a single batch of data with specific parameters"""
    audio_files, audio_dir, target_fs, clipping_threshold, time_clip, win_len, win_shift, delta, batch_id = batch_params
    
    print(f"Processing batch {batch_id}: fs={target_fs}, clip={clipping_threshold}, {len(audio_files)} files", flush=True)
    
    # Create a temporary directory for this batch
    temp_dir = f"temp_batch_{batch_id}"
    batch_audio_dir = os.path.join(os.getcwd(), temp_dir)
    os.makedirs(batch_audio_dir, exist_ok=True)
    
    # Copy files to temporary directory
    try:
        # Copy audio files to the temporary directory
        for audio_file in audio_files:
            # Get just the filename, not the path
            filename = os.path.basename(audio_file)
            dest_path = os.path.join(batch_audio_dir, filename)
            shutil.copy2(os.path.join(audio_dir, audio_file), dest_path)
        
        # Run training pipeline on the temporary directory
        training_data = training_data_func(
            audio_dir=batch_audio_dir,
            output_path=batch_audio_dir,  # Just a placeholder, not actually used for output
            target_fs_values=[target_fs],
            clipping_thresholds=[clipping_threshold],
            time_clip=[time_clip],
            win_len=win_len,
            win_shift=win_shift,
            delta=delta
        )
        
        print(f"Completed batch {batch_id}: fs={target_fs}, clip={clipping_threshold}", flush=True)
        return training_data
    
    finally:
        # Clean up - remove temporary directory
        shutil.rmtree(batch_audio_dir, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser(description="Generate training dataset with parallel processing")
    parser.add_argument("--audio_dir", type=str, required=True, help="Path to audio directory")
    parser.add_argument("--cnt", type=int, required=True, help="Number of files")
    parser.add_argument("--train_dir", type=str, required=True, help="Path to train directory")
    parser.add_argument("--test_dir", type=str, required=True, help="Path to test directory")
    parser.add_argument("--output_path", type=str, required=True, help="Path to the output directory")
    parser.add_argument("--target_fs_values", type=int, nargs='+', required=True, help="List of target sampling frequencies")
    parser.add_argument("--clipping_thresholds", type=float, nargs='+', required=True, help="List of clipping thresholds")
    parser.add_argument("--time_clip", type=int, nargs='+', required=True, help="List of time clipping values")
    parser.add_argument("--win_len", type=int, required=True, help="Window length")
    parser.add_argument("--win_shift", type=int, required=True, help="Window Shift")
    parser.add_argument("--delta", type=int, required=True, help="Delta value")
    parser.add_argument("--num_processes", type=int, default=6, help="Number of parallel processes to use")
    parser.add_argument("--num_batches", type=int, default=4, help="Number of batches to split the data into")
    parser.add_argument("--s_ratio", type=float, default=0.9, help="Split ratio")
    
    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_path, exist_ok=True)

    # Delete existing train and test directories if they exist
    for dir_path in [args.train_dir, args.test_dir]:
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)

    # Recreate train and test directories
    os.makedirs(args.train_dir, exist_ok=True)
    os.makedirs(args.test_dir, exist_ok=True)

    # Get list of all .wav files
    wav_files = [f for f in os.listdir(args.audio_dir) if f.endswith(".wav")]

    # Sample random subset if cnt is specified
    wav_files = random.sample(wav_files, min(args.cnt, len(wav_files)))

    # Shuffle files randomly
    random.shuffle(wav_files)

    # Split 90% train, 10% test
    split_idx = int(args.s_ratio * len(wav_files))
    train_files = wav_files[:split_idx]
    test_files = wav_files[split_idx:]

    # Copy files to respective directories
    for f in train_files:
        shutil.copy2(os.path.join(args.audio_dir, f), os.path.join(args.train_dir, f))

    for f in test_files:
        shutil.copy2(os.path.join(args.audio_dir, f), os.path.join(args.test_dir, f))

    print(f"Copied {len(train_files)} files to {args.train_dir}")
    print(f"Copied {len(test_files)} files to {args.test_dir}")

    # Split training files into batches
    total_files = len(train_files)
    batch_size = total_files // args.num_batches
    file_batches = []
    
    for i in range(args.num_batches - 1):
        file_batches.append(train_files[i*batch_size:(i+1)*batch_size])
    # Last batch includes any remaining files
    file_batches.append(train_files[(args.num_batches-1)*batch_size:])
    
    # Print batch sizes for verification
    for i, batch in enumerate(file_batches):
        print(f"Batch {i+1} size: {len(batch)} files", flush=True)
    
    # Prepare batch parameters
    batch_params = []
    batch_id = 1
    
    for target_fs in args.target_fs_values:
        for clipping_threshold in args.clipping_thresholds:
            for batch in file_batches:
                batch_params.append((
                    batch, 
                    args.train_dir, 
                    target_fs, 
                    clipping_threshold, 
                    args.time_clip[0],
                    args.win_len,
                    args.win_shift,
                    args.delta,
                    batch_id
                ))
                batch_id += 1
    
    # Process batches in parallel and collect all results directly
    print("Processing data in parallel...", flush=True)
    with multiprocessing.Pool(processes=args.num_processes) as pool:
        all_results = pool.map(process_batch, batch_params)
    
    # Combine all results into one list
    print("Combining all results...", flush=True)
    combined_training_data = []
    for result in all_results:
        combined_training_data.extend(result)
    
    # Save the combined data to a single pickle file
    combined_output_file = os.path.join(args.output_path, 'training_data.pkl')
    print(f"Saving combined data to {combined_output_file}", flush=True)
    
    with open(combined_output_file, 'wb') as f:
        pickle.dump(combined_training_data, f)
    
    print(f"Done! All data saved to {combined_output_file}", flush=True)
    print(f"Total training examples: {len(combined_training_data)}", flush=True)


if __name__ == "__main__":
    main()