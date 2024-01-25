import onnxruntime as ort
import numpy as np
import time
import csv
import librosa
from torchlibrosa.stft import Spectrogram, LogmelFilterBank
import torch
from torch import nn
from scipy.io import wavfile

#to import 
# audio_path = "MobileNetTens/helicopter.wav"
audio_path = "MobileNetTens/chant_oiseau.wav"
sample_rate = 32000

(waveform, _) = librosa.core.load(audio_path, sr=sample_rate, mono=True)
batch = waveform[None,0:96000]
print(batch.shape)

# np.save("chant_oiseau.npy", batch)

classes_name = np.load("MobileNetTens/classes.npy")
model_path = "MobileNetTens/Neural_Weight/LeeTens1D.onnx"

ort_sess = ort.InferenceSession(model_path)

start_time = time.time()
output = ort_sess.run(None, {'Input': batch})

print("\n--- %s seconds inference ---\r\n" % (time.time() - start_time))

# Print Result 
output = np.array(output[0][0])
print(output.shape)
sorted_indexes = np.argsort(output)[::-1]

top_k = 10  # Show top results
top_result_mat = output[sorted_indexes[0 : top_k]]    
top_classes = classes_name[sorted_indexes[0 : top_k]]
for i in range(top_k):
    print(i+1, "result : ", top_result_mat[i], " with class : ",top_classes[i],"\n")


