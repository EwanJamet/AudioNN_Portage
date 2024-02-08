#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import sys
sys.path.append('/home/xilinx')

# Needed to run inference on TCU
import time
import numpy as np
import pynq
from pynq import Overlay
from tcu_pynq.driver import Driver
from tcu_pynq.architecture import pynqz1

# Needed for unpacking and displaying image data
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
#from scipy.io import wavfile



# In[ ]:


overlay = Overlay('/home/xilinx/tensil_pynqz1.bit')
tcu = Driver(pynqz1, overlay.axi_dma_0)

fc_audioset_bias =  np.load('/home/xilinx/fc_audioset.bias.npy')
fc_audioset_weight = np.load('/home/xilinx/fc_audioset.weight.npy')
fc1_weight = np.load('/home/xilinx/fc1.weight.npy')
fc1_bias = np.load('/home/xilinx/fc1.bias.npy')

tcu.load_model('/home/xilinx/LeeTensJ96_9_2D_onnx_pynqz1.tmodel')


# In[ ]:


#(rate,sig) = wavfile.read('/home/xilinx/chant_oiseau.wav')

#batch = sig[0:96000,0]
#batch = batch.reshape(1,1,1,96000)
#print(batch.shape)

batch = np.load("/home/xilinx/helicopter.npy")
batch = batch.reshape(1,1,1,96000)


# In[ ]:


batch = np.zeros((1,1,1,96000))


# In[ ]:


def sigmoid(x):
    return 1/(1 + np.exp(-x))

inputs = {'Input': batch}
print(inputs["Input"])
print("input shape: ", inputs["Input"].shape)
start = time.time()
x = tcu.run(inputs)
end = time.time()
print("Ran inference in {:.4}s".format(end - start))
print()

np.savetxt("/home/xilinx/output",np.array(x["Output"]))

print("output shape: ", x["Output"].shape)
print("x= ",x["Output"])
start = time.time()

x = np.array(x["Output"]).reshape((1,256,4))
#print(x)

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

#print(x.shape)


# In[ ]:


# Print Result 

classes_name = np.load("/home/xilinx/classes.npy")
framewise_output = x[0]

sorted_indexes = np.argsort(framewise_output)[::-1]

top_k = 10  # Show top results
top_result_mat = framewise_output[sorted_indexes[0 : top_k]]    
top_classes = classes_name[sorted_indexes[0 : top_k]]
for i in range(top_k):
    print(i+1, "result : ", top_result_mat[i], " with class : ",top_classes[i],"\n")


# In[ ]:




