#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
sys.path.append('/home/xilinx')

# Needed to run inference on TCU
import time
import numpy as np
import pynq
from pynq import Overlay
from tcu_pynq.driver import Driver
from tcu_pynq.architecture import pynqz1

from collections import Counter
from collections import OrderedDict

# Needed for unpacking and displaying image data
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt

overlay = Overlay('/home/xilinx/tensil_pynqz1.bit')
tcu = Driver(pynqz1, overlay.axi_dma_0)

fc_audioset_bias =  np.load('/home/xilinx/fc_audioset.bias.npy')
fc_audioset_weight = np.load('/home/xilinx/fc_audioset.weight.npy')
fc1_weight = np.load('/home/xilinx/fc1.weight.npy')
fc1_bias = np.load('/home/xilinx/fc1.bias.npy')

tcu.load_model('/home/xilinx/LeeNet_onnx_pynqz1.tmodel')


# In[2]:


#Load l'audio de test
batch = np.load("/home/xilinx/helicopter.npy")
batch = batch[0][:59049]
batch = batch.reshape(1,1,1,59049)

def sigmoid(x):
    return 1/(1 + np.exp(-x))

inputs = {'Input': batch}


# In[4]:


# Boucle d'inférence et de calcul
top_indices = []
for i in range(10):
    
    # Inference section
    
    start = time.time()
    x = tcu.run(inputs)

    end = time.time()
    print("Ran inference in {:.4}s".format(end - start))
    print()

    
    # Calculations section

    start = time.time()

    x = np.array(x["Output"]).reshape((1,256,3))

    #max sur la dim 2 dans x1
    x1 = np.max(x, axis=2)
    #mean sur la dim 2 dans x2
    x2 = np.mean(x, axis=2)
    #x = x1 + x2
    x = x1 + x2
    x = x.reshape(1,256)
    #multiplication matricielle par fc1
    x = np.dot(x, np.transpose(fc1_weight)) + fc1_bias
    #ReLU de x
    x = np.maximum(0,x)
    #multiplication matricielle par fc audioset
    x = np.dot(x, np.transpose(fc_audioset_weight)) + fc_audioset_bias
    #sigmoid
    output = sigmoid(x)
    end = time.time()
    print("Ran calcul in {:.4}s".format(end - start))
    print()
    
    # Print Result 

    classes_name = np.load("/home/xilinx/classes.npy")
    framewise_output = x[0]

    sorted_indexes = np.argsort(framewise_output)[::-1]

    top_k = 10  # Show top results
    top_result_mat = framewise_output[sorted_indexes[0 : top_k]]
    top_indices = np.append(top_indices,sorted_indexes[0 : top_k])

    top_classes = classes_name[sorted_indexes[0 : top_k]]
    print("Results number ",i,"\n")
    for i in range(top_k):
        print(i+1, "result : ", top_result_mat[i], " with class : ",top_classes[i],"\n")

        
# Print final results
# Use Counter to count occurrences
element_counts = Counter(top_indices)

# Create a dictionary to store the results
result_dict = {element: count for element, count in element_counts.items()}

# Order the dictionary based on values
ordered_dict_values = OrderedDict(sorted(result_dict.items(), key=lambda x: x[1], reverse=True))

keys_array = list(ordered_dict_values.keys())
keys_array_integers = [int(key) for key in keys_array]

top_results_total = classes_name[keys_array_integers]
for i in range(len(top_results_total)):
        print(i+1, "e classe : ",top_results_total[i],"\n")


# In[3]:


# Print Result for one inference

classes_name = np.load("/home/xilinx/classes.npy")
framewise_output = x[0]

sorted_indexes = np.argsort(framewise_output)[::-1]

top_k = 10  # Show top results
top_result_mat = framewise_output[sorted_indexes[0 : top_k]]    
top_classes = classes_name[sorted_indexes[0 : top_k]]
for i in range(top_k):
    print(i+1, "result : ", top_result_mat[i], " with class : ",top_classes[i],"\n")


# In[ ]:




