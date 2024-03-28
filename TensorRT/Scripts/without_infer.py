########################################################

#### This script mesures the average power consumption over a pre-defined number of iterations WITHOUT inference for the purpose of precise measurements

############################################################
import os
import numpy as np
import time
import csv
import argparse
import librosa
from torchlibrosa.stft import Spectrogram, LogmelFilterBank
import torch
from torch import nn
import statistics
import pycuda.driver as cuda
import pycuda.autoinit
import tensorrt as trt

import matplotlib.pyplot as plt
plt.switch_backend('agg')
from threading import Thread
from jtop import jtop
import time
import  csv
from datetime import datetime, timedelta
from statistics import mean

# Global variables for controlling threads and tracking timing
thread_stop = False  # Boolean flag to control the threads
start_value = datetime.now()  # Global variable to store the current time
inference_start = datetime.now()  # Global variable to store the start time of inference
start_event_main = cuda.Event()  # CUDA event for tracking the main start event
start_time_infer = datetime.now()  # Global variable to store the start time of each inference
start_audio_tag = datetime.now()  # Global variable to store the start time of audio tagging
start_warmUp = datetime.now()  # Global variable to store the start time of GPU warm-up
end_warmpUp = datetime.now()  # Global variable to store the end time of GPU warm-up
elapsed_time = datetime.now()  # Global variable to store the elapsed time

inferences_time_list = []  # List to store inference times
inference_duration = []  # List to store inference durations


"""

 This function mesures the power consumption of the GPU, CV and the whole Jetson every 0.1s using the library jtop. It saves the values in a csv file and plot some graphs.

"""


def power_conso():
    global start_value, start_warmUp, start_audio_tag, result_folder
    global elapsed_time, thread_stop, end_warmpUp, inferences_time_list
    global inference_duration
    with jtop(interval=0.1) as jetson: # Open jtop for monitoring Jetson stats
    # jetson.ok() will provide the proper update frequency
        data={"Total Power" : [],"GPU" : [], "CV" : [], "Time" : []}
        while thread_stop==False:# Loop until thread_stop flag is True
            # Read Jetson stats and append to data dictionary
            data["Total Power"].append(jetson.stats['Power TOT'])
            data["GPU"].append(jetson.power["rail"]["GPU"]["power"])
            data["CV"].append(jetson.power["rail"]["CV"]["power"])
            data["Time"].append(jetson.stats['time'])
    start_value = data["Time"][0]
    data["Time"] = [(time_value - start_audio_tag).total_seconds() for time_value in data["Time"] ]

    # Calculate mean power consumption for GPU and CV
    comparison_time_inf = np.array(data["Time"]) > inferences_time_list[0]
    comparison_time_sup = np.array(data["Time"]) < inferences_time_list[1]
    time_cond = comparison_time_inf & comparison_time_sup
    mean_conso_GPU = np.mean(np.array(data["GPU"])[time_cond])
    mean_conso_CV = np.mean(np.array(data["CV"])[time_cond])

    # Plot GPU power consumption
    plt.plot(data["Time"], data["GPU"], label='GPU')
    plt.xlabel("Time (s)")
    plt.ylabel("Power Consumption GPU (mW)")
    plt.annotate(" Mean GPU Consumption:" + str(round(mean_conso_GPU,5)) + " (mW)\n Mean CV Consumption: " + str(round(mean_conso_CV,5)) + " (mW)", xy=(0.05,0.95), xycoords='axes fraction',ha='left', va='top', bbox=dict(boxstyle='round',alpha=0.3, facecolor='white'), fontsize='small')

    # Add vertical lines for inference times
    for inf in inferences_time_list:
    	plt.axvline(x=inf, color='red')

    # Save GPU power consumption plot
    plt.savefig(os.path.join(os.getcwd(),'Results',result_folder,'Trt_Jetson_GPU_Consumption_rep_1000.png'))

    # Plot CV power consumption
    plt.plot(data["Time"], data["CV"], label='CV')
    plt.legend()
    plt.xlabel("Time (s)")
    plt.ylabel("Power Consumption CV (mW)")
    plt.savefig(os.path.join(os.getcwd(),'Results',result_folder,'Trt_Jetson_CV_Consumption_rep_1000.png'))

    # Write data to CSV file
    with open(os.path.join(os.getcwd(),'Results',result_folder,'power_conso.csv'), 'w', newline='') as f:
    	writer = csv.writer(f)
    	writer.writerow(data.keys())
    	writer.writerows(zip(*data.values()))





"""
    This function tags the same audio nb_rep times.
    In:
        * output_data:   Empty array of the size of the output.
        * nb-rep:        Number of repetitions.

    The audio is passed as an argument in the trt_inference script. It is a spectrogram of the audio.
"""

# Function for performing audio tagging without inference
def audio_tagging(output_data, nb_rep):

	global start_event_main, start_audio_tag, start_warmUp, end_warmpUp
	start_event_main.record()
	start_audio_tag = datetime.now()

	input_data = np.load(os.path.abspath(os.path.join(os.getcwd(),args.audio_path)))

	# Load trt engine file
	model_path_trt = os.path.join(os.getcwd(),args.engine_path)
	with open(model_path_trt, 'rb') as f:
		  engine_data = f.read()

	runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
	engine = runtime.deserialize_cuda_engine(engine_data)
	context = engine.create_execution_context()

	output_shape = (526,) # Output shape

	# GPU Warm Up
	start_warmUp=(datetime.now() - start_audio_tag).total_seconds()
	dummy_input = np.random.rand(1,224000)
	dummy_output = np.empty(output_shape, dtype=np.float32)
	dummy_input_buf = cuda.mem_alloc(dummy_input.nbytes)
	cuda.memcpy_htod_async(dummy_input_buf, dummy_input)
	dummy_output_buf = cuda.mem_alloc(output_shape[0] * np.dtype(np.float32).itemsize)
	context.execute_v2(bindings=[int(dummy_input_buf),int(dummy_output_buf)])
	cuda.memcpy_dtoh_async(dummy_output, dummy_output_buf)
	end_warmpUp=(datetime.now() - start_audio_tag).total_seconds()

	# Allocate input and output buffers on the GPU :
	input_buf = cuda.mem_alloc(input_data.nbytes)
	output_buf = cuda.mem_alloc(output_shape[0] * np.dtype(np.float32).itemsize)

	# Transfer input data to GPU
	cuda.memcpy_htod_async(input_buf, input_data)

	# Inference Time Variables & Start Recording :
	global start_value, inference_start, start_time_infer, inferences_time_list
	global inference_duration
  	global elapsed_time
  	global thread_stop
	start_infer=cuda.Event()
	end_event=cuda.Event()

	inference_start= (datetime.now() - start_value).total_seconds()

	start_infer.record()
	start_time_infer=start_event_main.time_till(start_infer)/1000
	inference_start= (datetime.now() - start_value).total_seconds()
	inferences_time_list.append(inference_start)

	start_one_infer=cuda.Event()
	end_one_event=cuda.Event()

	inference_duration=[]
	for i in range(nb_rep):
	    start_one_infer.record()
	    end_one_event.record()

	end_event.record()
	elapsed_time=start_infer.time_till(end_event)/1000
	inference_end= (datetime.now() - start_value).total_seconds()
	inferences_time_list.append(inference_end)
  # Transfer output data from GPU :
	cuda.memcpy_dtoh_async(output_data, output_buf)

	# Stop Consumption Recording :
	time.sleep(5)
	thread_stop=True



if __name__ == '__main__':

    # parse arguments
    parser = argparse.ArgumentParser(description='Arguments specified in the text file')
    parser.add_argument('--engine_path', required=True)
    parser.add_argument('--audio_path', required=True)
    parser.add_argument('--engine', required=True)
    parser.add_argument('--power_mode', required=True)
    parser.add_argument('--nb_rep', required=True)

    nb_rep=int(args.nb_rep)
    args = parser.parse_args()

    global result_folder
    result_folder = str(datetime.now().strftime("%Y-%m-%d %H:%M"))

    os.mkdir(os.path.join(os.getcwd(),'Results',result_folder))
    print("path", os.path.join(os.getcwd(),'Results',result_folder))

    classes_name = np.load(os.path.join(os.getcwd(),'Sources','classes.npy'))
    output_shape = (len(classes_name)-1,)
    output_data = np.empty(output_shape, dtype=np.float32)
    end_event_main=cuda.Event()
    #global start_event_main
    #start_event_main.record()
    thread_conso=Thread(target = power_conso)
    thread_conso.start()
    thread_model=Thread(target = audio_tagging, args=(output_data,))
    end_event_main.record()

    # Print Result
    audio_tagging(output_data)

    framewise_output = output_data
    sorted_indexes = np.argsort(framewise_output)[::-1]

    top_k = 10  # Show top results
    top_result_mat = framewise_output[sorted_indexes[0 : top_k]]
    top_classes = classes_name[sorted_indexes[0 : top_k]]
    for i in range(top_k):
        print(i+1, "result : ", top_result_mat[i], " with class : ",top_classes[i],"\n")

    # Store parameters
    with open(os.path.join(os.getcwd(),'Results',result_folder,'parameters'),'w') as param:
        param.write("Engine : ")
        param.write(args.engine)
        param.write("\n")
        param.write("Power Mode : ")
        param.write(args.power_mode)

    time.sleep(20)


