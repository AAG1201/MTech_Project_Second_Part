import numpy as np
from tqdm import tqdm
from ml_aspade import ml_aspade_train
from ml_aspade import ml_aspade_eval
from gabwin import gabwin
from gabdual import gabdual
from peak_normalize import peak_normalize

def spade_segmentation_train(clipped_signal, resampled_data, Ls, win_len, win_shift, ps_maxit, ps_epsilon, ps_r, ps_s, F_red, masks):
  
  win_len = int(win_len)
  win_shift = int(win_shift)
  # Implement the SPADE algorithm for reconstruction
  L = int(np.ceil(Ls / win_shift) * win_shift + (np.ceil(win_len / win_shift) - 1) * win_shift) # L is divisible by a and minimum amount of zeros equals gl (window length). Zeros will be appended to avoid periodization of nonzero samples.
  N = L // win_shift

  # padding the signals and masks to length L
  padding = np.zeros(int(L - Ls))
  data_clipped = np.concatenate([clipped_signal, padding])
  data_orig = np.concatenate([resampled_data, padding])
  masks['Mr'] = np.concatenate([masks['Mr'], np.ones(int(L - Ls), dtype=bool)])
  masks['Mh'] = np.concatenate([masks['Mh'], np.zeros(int(L - Ls), dtype=bool)])
  masks['Ml'] = np.concatenate([masks['Ml'], np.zeros(int(L - Ls), dtype=bool)])

  # Construction of analysis and synthesis windows
  g = gabwin(win_len)
  gana = peak_normalize(g)  # Peak-normalization of the analysis window
  gsyn = gabdual(gana, win_shift, win_len) * win_len  # Computing the synthesis window

  # This is substituting fftshift (computing indexes to swap left and right half of the windows)
  idxrange = np.concatenate([np.arange(0, np.ceil(win_len / 2)), np.arange(-np.floor(win_len / 2), 0)])
  idxrange2 = idxrange + abs(np.min(idxrange))

  # Convert the float array to integer array
  idxrange = idxrange.astype(int)
  idxrange2 = idxrange2.astype(int)

  # Initialization of signal blocks
  data_block = np.zeros(win_len)
  data_orig_block = np.zeros(win_len)
  data_rec_fin = np.zeros(L)

  # initialization of parameters for one signal block
  ps_Ls = win_len
  masks_seg = {
    'Mr': np.ones(win_len).astype(bool),
    'Mh': np.zeros(win_len).astype(bool),
    'Ml': np.zeros(win_len).astype(bool)
  }

  k_final = len(data_block)

  training_data = []

  # Main loop
  # for n in tqdm(range(N), desc="Processing", unit="iteration"):
  for n in range(N):
    # multiplying signal block with windows and choosing corresponding masks
    idx = np.mod(n * win_shift + idxrange, L)
    idx = idx.astype(int)
    data_block[idxrange2] = data_clipped[idx] * gana
    data_orig_block[idxrange2] = data_orig[idx] * gana

    assert len(masks['Mr']) > np.max(idx), "Index 'idx' exceeds masks['Mr'] dimensions"
    assert len(masks_seg['Mr']) > np.max(idxrange2), "Index 'idxrange2' exceeds masks_seg['Mr'] dimensions"

    masks_seg['Mr'][idxrange2] = masks['Mr'][idx]
    masks_seg['Mh'][idxrange2] = masks['Mh'][idx]
    masks_seg['Ml'][idxrange2] = masks['Ml'][idx]

    data_rec_block, metrics, cycles = ml_aspade_train(data_block, masks_seg, ps_Ls, ps_maxit, ps_epsilon, ps_r, ps_s, F_red)

    # Folding blocks together using Overlap-Add approach (OLA)
    data_rec_block = np.fft.ifftshift(data_rec_block)
    data_rec_fin[idx] = data_rec_fin[idx] + data_rec_block * gsyn
    training_data.append([metrics['initial_estimate'], metrics['best_estimate'], metrics['best_sparsity']]) 

  data_rec_fin = data_rec_fin[:Ls]

  return data_rec_fin, metrics, training_data, cycles


def spade_segmentation_eval(clipped_signal, resampled_data, Ls, win_len, win_shift, ps_maxit, ps_epsilon, ps_r, ps_s, F_red, masks, model_path, loaded_model, device):

  win_len = int(win_len)
  win_shift = int(win_shift)
  # Implement the SPADE algorithm for reconstruction
  L = int(np.ceil(Ls / win_shift) * win_shift + (np.ceil(win_len / win_shift) - 1) * win_shift)
  # L is divisible by a and minimum amount of zeros equals gl (window length).
  # Zeros will be appended to avoid periodization of nonzero samples.
  N = L // win_shift

  # padding the signals and masks to length L
  padding = np.zeros(int(L - Ls))
  data_clipped = np.concatenate([clipped_signal, padding])
  data_orig = np.concatenate([resampled_data, padding])
  masks['Mr'] = np.concatenate([masks['Mr'], np.ones(int(L - Ls), dtype=bool)])
  masks['Mh'] = np.concatenate([masks['Mh'], np.zeros(int(L - Ls), dtype=bool)])
  masks['Ml'] = np.concatenate([masks['Ml'], np.zeros(int(L - Ls), dtype=bool)])

  # Construction of analysis and synthesis windows
  g = gabwin(win_len)
  gana = peak_normalize(g)  # Peak-normalization of the analysis window
  gsyn = gabdual(gana, win_shift, win_len) * win_len  # Computing the synthesis window

  # This is substituting fftshift (computing indexes to swap left and right half of the windows)
  idxrange = np.concatenate([np.arange(0, np.ceil(win_len / 2)), np.arange(-np.floor(win_len / 2), 0)])
  idxrange2 = idxrange + abs(np.min(idxrange))

  # Convert the float array to integer array
  idxrange = idxrange.astype(int)
  idxrange2 = idxrange2.astype(int)

  # Initialization of signal blocks
  data_block = np.zeros(win_len)
  data_orig_block = np.zeros(win_len)
  data_rec_fin = np.zeros(L)

  # initialization of parameters for one signal block
  ps_Ls = win_len
  masks_seg = {
    'Mr': np.ones(win_len).astype(bool),
    'Mh': np.zeros(win_len).astype(bool),
    'Ml': np.zeros(win_len).astype(bool)
  }

  # Main loop
  for n in tqdm(range(N), desc="Processing", unit="iteration"):
    # multiplying signal block with windows and choosing corresponding masks
    idx = np.mod(n * win_shift + idxrange, L)
    idx = idx.astype(int)
    data_block[idxrange2] = data_clipped[idx] * gana
    data_orig_block[idxrange2] = data_orig[idx] * gana

    assert len(masks['Mr']) > np.max(idx), "Index 'idx' exceeds masks['Mr'] dimensions"
    assert len(masks_seg['Mr']) > np.max(idxrange2), "Index 'idxrange2' exceeds masks_seg['Mr'] dimensions"

    masks_seg['Mr'][idxrange2] = masks['Mr'][idx]
    masks_seg['Mh'][idxrange2] = masks['Mh'][idx]
    masks_seg['Ml'][idxrange2] = masks['Ml'][idx]

    data_rec_block, cycles = ml_aspade_eval(data_block, masks_seg, ps_Ls, ps_maxit, ps_epsilon, ps_r, ps_s,F_red, model_path, loaded_model, device)

    # Folding blocks together using Overlap-Add approach (OLA)
    data_rec_block = np.fft.ifftshift(data_rec_block)
    data_rec_fin[idx] = data_rec_fin[idx] + data_rec_block * gsyn

  data_rec_fin = data_rec_fin[:Ls]

  return data_rec_fin, cycles


