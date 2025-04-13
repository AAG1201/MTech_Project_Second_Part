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


def training_data_func(audio_dir: str,
                      output_path: str,
                      target_fs_values: List[int],
                      clipping_thresholds: List[float],
                      time_clip: List[int],
                      K: int,
                      delta: int
                      ):
    """
    Complete training pipeline for ASPADE ML enhancement
    """
    
    training_data = []
    
    total_combinations = len(target_fs_values) * len(clipping_thresholds) * len(time_clip)
    # pbar = tqdm(total=total_combinations, desc="Processing configurations")
    
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

                    # data = data[fs * tc : 2 * fs * tc]
                    
                    data = data / max(np.abs(data))  
                    
                    resampled_data = resample(data, int(target_fs * tc))
                    
                    # Setup parameters
                    Ls = len(resampled_data)
                    win_len = np.floor(Ls/K)
                    win_shift = np.floor(win_len/4)
                    F_red = 2
                    
                    # ASPADE parameters
                    ps_s = 1
                    ps_r = 2
                    ps_epsilon = 0.1
                    ps_maxit = np.ceil(np.floor(win_len * F_red / 2 + 1) * ps_r / ps_s)

                    
                    # Generate clipped signal
                    # print("Generating clipped signal...")
                    clipped_signal, masks, theta, true_sdr, percentage = clip_sdr_modified(resampled_data, clipping_threshold)
                    # print(f"Clipping stats - Threshold: {theta:.3f}, SDR: {true_sdr:.2f} dB, "
                    #       f"Clipped: {percentage:.2f}%, Duration: {tc}sec")
                    
                    # Reconstruction and timing
                    # print("Performing reconstruction...")
                    start_time = time()
                    reconstructed_signal, metrics, intermediate_training_data = spade_segmentation_train(
                        clipped_signal, resampled_data, Ls, win_len, win_shift,
                        ps_maxit, ps_epsilon, ps_r, ps_s, F_red, masks
                    )
                    elapsed_time = time() - start_time

                    reconstructed_signal = resample(reconstructed_signal, int(fs * tc))

                    # Calculate metrics
                    sdr_reconstructed = sdr(data, reconstructed_signal)
                    sdr_improvement = sdr_reconstructed - true_sdr
                    # print(f"Reconstruction time: {elapsed_time:.2f}s")
                    print(f"Done with SDR imp:{sdr_improvement:.2f} dB and {percentage:.2f}% and {true_sdr:.2f} dB", flush=True)

                    training_data.extend(intermediate_training_data)

                
                # pbar.update(1)
    
    # pbar.close()
    
    return training_data

  


def main():
    parser = argparse.ArgumentParser(description="Generate training dataset")
    parser.add_argument("--audio_dir", type=str, required=True, help="Path to audio directory")
    parser.add_argument("--cnt", type=int, required=True, help="Number of files")
    parser.add_argument("--train_dir", type=str, required=True, help="Path to train directory.")
    parser.add_argument("--test_dir", type=str, required=True, help="Path to test directory.")
    parser.add_argument("--output_path", type=str, help="Path to the output directory.")
    parser.add_argument("--target_fs_values", type=int, nargs='+', help="List of target sampling frequencies.")
    parser.add_argument("--clipping_thresholds", type=float, nargs='+', help="List of clipping thresholds.")
    parser.add_argument("--time_clip", type=int, nargs='+', help="List of time clipping values.")
    parser.add_argument("--K", type=int, required=True, help="K value")
    parser.add_argument("--delta", type=int, required=True, help="Delta value")
    
    
    args = parser.parse_args()


    # Delete existing train and test directories if they exist
    for dir_path in [args.train_dir, args.test_dir]:
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)

    # Recreate train and test directories
    os.makedirs(args.train_dir, exist_ok=True)
    os.makedirs(args.test_dir, exist_ok=True)

    # Get list of all .wav files
    wav_files = [f for f in os.listdir(args.audio_dir) if f.endswith(".wav")]

    wav_files = random.sample(wav_files, min(args.cnt, len(wav_files)))

    # Shuffle files randomly
    random.shuffle(wav_files)

    # Split 80% train, 20% test
    split_idx = int(0.9 * len(wav_files))
    train_files = wav_files[:split_idx]
    test_files = wav_files[split_idx:]

    # Copy files to respective directories
    for f in train_files:
        shutil.copy2(os.path.join(args.audio_dir, f), os.path.join(args.train_dir, f))

    for f in test_files:
        shutil.copy2(os.path.join(args.audio_dir, f), os.path.join(args.test_dir, f))

    print(f"Copied {len(train_files)} files to {args.train_dir}")
    print(f"Copied {len(test_files)} files to {args.test_dir}")



    # Run training pipeline
    training_data = training_data_func(
        audio_dir=args.train_dir,
        output_path=args.output_path,
        target_fs_values=args.target_fs_values,
        clipping_thresholds=args.clipping_thresholds,
        time_clip=args.time_clip,
        K=args.K,
        delta=args.delta
    )

    # Save using pickle
    with open(os.path.join(args.output_path, 'training_data.pkl'), 'wb') as f:
        pickle.dump(training_data, f)  

if __name__ == "__main__":
    main()