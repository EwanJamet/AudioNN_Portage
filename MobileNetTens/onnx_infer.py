import onnxruntime as ort
import numpy as np
import time
import csv


#to import 

x = np.load('image_701x64.npy') # attention ! à modifier en cas de changement de fichier 
classes_name = np.load("classes.npy")
model_path = "/home/ewan/Desktop/audioset_tagging_cnn/MobileNetTens/MobileNetTens_W.onnx"

ort_sess = ort.InferenceSession(model_path)

start_time = time.time()
outputs = ort_sess.run(None, {'Input': x})

print("\n--- %s seconds inference ---\r\n" % (time.time() - start_time))

# Print Result 


framewise_output = outputs[0][0]

sorted_indexes = np.argsort(framewise_output)[::-1]

top_k = 10  # Show top results
top_result_mat = framewise_output[sorted_indexes[0 : top_k]]    
top_classes = classes_name[sorted_indexes[0 : top_k]]
for i in range(top_k):
    print(i+1, "result : ", top_result_mat[i], " with class : ",top_classes[i],"\n")


