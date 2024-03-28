import onnxruntime as ort
import numpy as np
import time
import csv
import librosa
from torchlibrosa.stft import Spectrogram, LogmelFilterBank
import torch
from torch import nn
import os

OUTPUT_DIRECTORY = os.path.dirname(os.path.realpath(__file__))

fc_audioset_bias =  np.load(OUTPUT_DIRECTORY + '/Input/weight_leeTens/fc_audioset.bias.npy')
fc_audioset_weight = np.load(OUTPUT_DIRECTORY +'/Input/weight_leeTens/fc_audioset.weight.npy')
fc1_weight = np.load(OUTPUT_DIRECTORY +'/Input/weight_leeTens/fc1.weight.npy')
fc1_bias = np.load(OUTPUT_DIRECTORY+ '/Input/weight_leeTens/fc1.bias.npy')

batch = np.load(OUTPUT_DIRECTORY + "/Input/helicopter.npy")
batch = batch[0][:59049]
batch = batch.reshape(1,1,1,59049)

# batch = np.array((1,1,1,batch),dtype=np.float32)


classes_name = np.load(OUTPUT_DIRECTORY + "/BaseModel/classes.npy")


ort_sess = ort.InferenceSession(OUTPUT_DIRECTORY + "/Model_ONNX/LeeNet.onnx")
output = np.array(ort_sess.run(None, {'Input': batch}))

def sigmoid(x):
    return 1/(1 + np.exp(-x))

start = time.time()

x = np.array(output).reshape((1,256,3))

#max sur la dim 2 dans x1
x1 = np.max(x, axis=2)
#mean sur la dim 2 dans x2
x2 = np.mean(x, axis=2)
#x = x1 + x2
x = x1 + x2
#print(x.shape)
x = x.reshape(1,256)
#multiplication matriciel par fc1
x = np.dot(x, np.transpose(fc1_weight)) + fc1_bias
#ReLU de x
x = np.maximum(0,x)
#multiplication matriciel par fc audioset
x = np.dot(x, np.transpose(fc_audioset_weight)) + fc_audioset_bias
#sigmoid
output = sigmoid(x)
end = time.time()
print("Ran calcul in {:.4}s".format(end - start))
print()

# Print Result 


framewise_output = x[0]

sorted_indexes = np.argsort(framewise_output)[::-1]

top_k = 10  # Show top results
top_result_mat = framewise_output[sorted_indexes[0 : top_k]]    
top_classes = classes_name[sorted_indexes[0 : top_k]]
for i in range(top_k):
    print(i+1, "result : ", top_result_mat[i], " with class : ",top_classes[i],"\n")







