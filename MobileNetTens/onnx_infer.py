import onnxruntime as ort
import numpy as np
import time
import csv

import librosa
from torchlibrosa.stft import Spectrogram, LogmelFilterBank
import torch
from torch import nn

audio_path = "resources/R9_ZSCveAHg_7s.wav"
sample_rate = 32000

(waveform, _) = librosa.core.load(audio_path, sr=sample_rate, mono=True)
waveform = waveform[None, :]    # (1, audio_length)

sample_rate =32000
window_size = 1024
hop_size =320
mel_bins =64
fmin =  50     
fmax =14000
window = 'hann'
center = True
pad_mode = 'reflect'
ref = 1.0
amin = 1e-10
top_db = None

start_time = time.time()
t_waveform = torch.from_numpy(waveform)
spectrogram_extractor = Spectrogram(n_fft=window_size, hop_length=hop_size, 
            win_length=window_size, window=window, center=center, pad_mode=pad_mode, 
            freeze_parameters=True)

x = spectrogram_extractor(t_waveform)

logmel_extractor = LogmelFilterBank(sr=sample_rate, n_fft=window_size, 
            n_mels=mel_bins, fmin=fmin, fmax=fmax, ref=ref, amin=amin, top_db=top_db, 
            freeze_parameters=True)


x = logmel_extractor(x)


model_path = "/home/ewan/Desktop/audioset_tagging_cnn/MobileNetTens/MobileNetTens_W.onnx"

ort_sess = ort.InferenceSession(model_path)

start_time = time.time()
outputs = ort_sess.run(None, {'Input': x.numpy()})
print("--- %s seconds inference ---" % (time.time() - start_time))

# Print Result 
label_path = "/home/ewan/Desktop/audioset_tagging_cnn/metadata/class_labels_indices.csv" 


framewise_output = outputs[0][0]

sorted_indexes = np.argsort(framewise_output)[::-1]

top_k = 10  # Show top results
top_result_mat = framewise_output[sorted_indexes[0 : top_k]]    

print(top_result_mat)



