# List 

# To copy files of duration atleats 2s
# Copy files from one dir to other 
# Data Generation
# To check number of files
# Normal declipping audio
# GIT PUSH

###############################################################################################################################

# To copy files of duration atleats 2s

import os
import shutil
import wave

# Source and destination directories
src_dir = '/data2/AAG/MTech_Project_Data/speech_data'
dst_dir = '/data2/AAG/MTech_Project_Data/speech_data_filter'

# Create destination directory if it doesn't exist
os.makedirs(dst_dir, exist_ok=True)

# Minimum duration in seconds
min_duration = 2.0

# Go through each file in the source directory
for file_name in os.listdir(src_dir):
    if file_name.lower().endswith('.wav'):
        src_path = os.path.join(src_dir, file_name)
        try:
            with wave.open(src_path, 'rb') as wf:
                frames = wf.getnframes()
                rate = wf.getframerate()
                duration = frames / float(rate)
                
                if duration >= min_duration:
                    dst_path = os.path.join(dst_dir, file_name)
                    shutil.copy2(src_path, dst_path)
                    print(f"Copied: {file_name} ({duration:.2f}s)")
        except wave.Error as e:
            print(f"Skipping {file_name} due to error: {e}")

###############################################################################################################################

# Copy files from one dir to other 

import os
import shutil

# Source and destination directories
src_dir = '/data2/AAG/amartya_sounds'
dst_dir = '/data2/AAG/MTech_Project_Data/speech_data'

# Create destination directory if it doesn't exist
os.makedirs(dst_dir, exist_ok=True)

# Walk through all directories and subdirectories
for root, dirs, files in os.walk(src_dir):
    for file in files:
        if file.lower().endswith('.wav'):
            src_path = os.path.join(root, file)
            dst_path = os.path.join(dst_dir, file)
            
            # To avoid overwriting if duplicate filenames exist
            base, ext = os.path.splitext(file)
            count = 1
            while os.path.exists(dst_path):
                dst_path = os.path.join(dst_dir, f"{base}_{count}{ext}")
                count += 1

            shutil.copy2(src_path, dst_path)
            print(f"Copied: {src_path} -> {dst_path}")


###############################################################################################################################


# Data Generation

# nohup python training_data_gen_new.py --audio_dir speech_data --cnt 2000 --train_dir train_data --test_dir test_data --output_path pkl_data --target_fs_values 16000 --clipping_thresholds 0.1 0.2 --time_clip 1 --win_len 500 --win_shift 125 --delta 300 --s_ratio 0.9 > training_log.txt 2>&1 &


###############################################################################################################################


# To check number of files

# find /data2/AAG/MTech_Project_Speech/speech_data -type f -name "*.wav" | wc -l

###############################################################################################################################

# Normal declipping audio


# win_len = 250
# win_shift = int(win_len / 4)

# !python process.py --audio_dir custom_sound \
#     --output_path output_sound \
#     --time_clip 1 \
#     --target_fs_values 16000 \
#     --clipping_thresholds 0.2 \
#     --dynamic 1 \
#     --saving 0 \
#     --plotting 0 \
#     --delta 0 \
#     --win_len {win_len} \
#     --win_shift {win_shift}

###############################################################################################################################


# GIT PUSH

# cd /data2/AAG/MTech_Project_Second_Part
# git init
# echo -e "*.wav\n*.pkl\n*.pth\n*.log\naagproj/\n__pycache__/\n*.pyc\n.env\n.DS_Store" > .gitignore
# cat .gitignore
# git reset
# git add .
# git commit -m "Initial backup (excluding .wav files)"
# git remote add origin https://github.com/AAG1201/MTech_Project_Second_Part.git
# git branch -M main
# git push -u origin main