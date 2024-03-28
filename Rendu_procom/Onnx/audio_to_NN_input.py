import librosa
from torchlibrosa.stft import Spectrogram, LogmelFilterBank
import torch
from torch import nn
import numpy as np
import time

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
print("\n--- %s seconds conversion from audio to spectrogram ---\r\n" % (time.time() - start_time))

x_name=x.numpy()[0][0]
name_file= 'image_' + str(x_name.shape[0]) + "x64.npy"

np.save(name_file, x.numpy())


#need just to extract the x value, dim : torch.Size([1, 1, lenght_audio , 64 (the only dimension that matter which is the spectrogram)])
#x is a tensor maybe you need to convert it in a np.array to be readable by the program 
# to convert it just use numpy_array = x.numpy()
