
import numpy as np
import time
import csv

import librosa
from torchlibrosa.stft import Spectrogram, LogmelFilterBank
import torch
from torch import nn


import pycuda.driver as cuda
import pycuda.autoinit
import tensorrt as trt


from threading import Thread
from jtop import jtop
import time
import  csv
from datetime import datetime, timedelta


thread_stop=False
start_value=datetime.now()
inference_start=datetime.now()


def power_conso():
    with jtop() as jetson:
    # jetson.ok() will provide the proper update frequency
        global thread_stop
        data={"Total Power" : [], "Time":[]}
        while jetson.ok() and thread_stop==False:
            # Read tegra stats
            data["Total Power"].append(jetson.stats['Power TOT'])
            data["Time"].append(jetson.stats['time'])
    global start_value
    global inference_start
    start_value = data["Time"][0]
    data["Time"] = [(time_value - start_value).total_seconds() for time_value in data["Time"] ]
    
    print(data)
    #lists = sorted(data.items())
    #x,y = zip(*lists)
    #pl.plot(x,y)
    plt.plot(data["Time"], data["Total Power"])
    plt.xlabel("Time (s)")
    plt.ylabel("Power Consumption (mW)")
    plt.axvline(x=inference_start, color='red')
    plt.savefig("/home/brain/Documents/Test/audioset_tagging_cnn/TensorRT/results/Jetson_Power_Consumption.png")
    
    with open("/home/brain/Documents/Test/audioset_tagging_cnn/TensorRT/results/power_conso.csv", 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(data.keys())
        writer.writerows(zip(*data.values()))
    
    
    
def audio_tagging(output_data):
	input_data= np.load('/home/brain/Documents/AudioNN_Portage/MobileNetTens/image_701x64.npy') # attention ! à modifier en cas de changement de fichier 
	

	# Load trt engine file
	model_path_trt = "/home/brain/Documents/Test/audioset_tagging_cnn/TensorRT/mobileNet_engine.trt"

	with open(model_path_trt, 'rb') as f:
		engine_data = f.read()
		
	runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
	engine = runtime.deserialize_cuda_engine(engine_data)
	context = engine.create_execution_context()

		
	# Allocate input and output buffers on the GPU
	input_buf = cuda.mem_alloc(input_data.nbytes)
	print(input_buf)


	output_shape = (526,)  # Example output shape
	output_buf = cuda.mem_alloc(output_shape[0] * np.dtype(np.float32).itemsize)

	# Transfer input data to GPU
	cuda.memcpy_htod(input_buf, input_data)

	# Run inference
	context.execute(bindings=[int(input_buf),int(output_buf)])

	# Transfer output data from GPU
	
	cuda.memcpy_dtoh(output_data, output_buf)
    
    global thread_stop
    time.sleep(10)
    thread_stop=True
	


	

if __name__ == '__main__':

    output_shape = (526,)  # Example output shape
    output_data = np.empty(output_shape, dtype=np.float32)

    thread_conso=Thread(target = power_conso)
    thread_conso.start()
    thread_model=Thread(target = audio_tagging, args=(output_data,))
    thread_model.start()

    # Print Result 

    audio_tagging(output_data)
    classes_name = np.load("/home/brain/Documents/AudioNN_Portage/MobileNetTens/classes.npy")

    
    framewise_output = output_data
    sorted_indexes = np.argsort(framewise_output)[::-1]

    top_k = 10  # Show top results
    top_result_mat = framewise_output[sorted_indexes[0 : top_k]]    
    top_classes = classes_name[sorted_indexes[0 : top_k]]
    for i in range(top_k):
        print(i+1, "result : ", top_result_mat[i], " with class : ",top_classes[i],"\n")
