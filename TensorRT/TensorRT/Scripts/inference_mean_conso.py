############################################################

#### This script mesures the average inference time and consumption over a pre-defined number of iterations.

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

# Definition of the number of repetitions of inferences the values are computed over
NB_REP = 10000

# thread_stop = True to stop the measurments
thread_stop=False

# Definition of the global variables required to mesure time
start_value=datetime.now()
inference_start=datetime.now()
start_event_main=cuda.Event()
start_time_infer=datetime.now()
start_audio_tag=datetime.now()
start_warmUp=datetime.now()
end_warmpUp=datetime.now()

# Store the inference time of each repetition
inferences_time_list=[]
inference_duration=[]


"""

 This function mesures the energy consumption of the GPU, CV and the whole Jetson every 0.1s using the library jtop. It saves the values in a csv file and plot some graphs.

"""

def power_conso():
    global start_value
    global start_warmUp
    global start_audio_tag
    global result_folder
    with jtop(interval=0.1) as jetson:
    # jetson.ok() will provide the proper update frequency
        global thread_stop
        global end_warmpUp
        data={"Total Power" : [],"GPU" : [], "CV" : [], "TOT" : [], "Time" : []}
        while thread_stop==False:
            # Read tegra stats
            data["Total Power"].append(jetson.stats['Power TOT'])
            data["GPU"].append(jetson.power["rail"]["GPU"]["power"])
            data["CV"].append(jetson.power["rail"]["CV"]["power"])
            data["TOT"].append(jetson.power["tot"]["power"])
            data["Time"].append(jetson.stats['time'])
    global inferences_time_list
    start_value = data["Time"][0]
    data["Time"] = [(time_value - start_audio_tag).total_seconds() for time_value in data["Time"] ]

    # mean
    comparison_time_inf = np.array(data["Time"]) > inferences_time_list[0]
    comparison_time_sup = np.array(data["Time"]) < inferences_time_list[1]
    time_cond = comparison_time_inf & comparison_time_sup
    mean_conso_GPU = np.mean(np.array(data["GPU"])[time_cond])
    mean_conso_CV = np.mean(np.array(data["CV"])[time_cond])
    mean_conso_tot = np.mean(np.array(data["TOT"])[time_cond])
    global inference_duration
    print("|",args.power_mode, args.engine,  round(mean(inference_duration),6), round(mean_conso_GPU,4), round(mean_conso_CV,4), round(mean_conso_tot,4))


    plt.plot(data["Time"], data["GPU"], label='GPU')
    plt.xlabel("Time (s)")
    plt.ylabel("Power Consumption GPU (mW)")

    plt.title(label =" Mean inference time: "+str(round(mean(inference_duration),5)) + " (s) | Mean GPU :" + str(round(mean_conso_GPU,5)) + " (mW) | Mean CV : " + str(round(mean_conso_CV,5)) + " (mW)", fontsize=8)
    for inf in inferences_time_list:
    	plt.axvline(x=inf, color='red')
    plt.savefig(os.path.join(os.getcwd(),'Results',result_folder,'Trt_Jetson_GPU_Consumption_rep_1000.png'))

    plt.clf()

    plt.plot(data["Time"], data["CV"], label='CV')
    #plt.plot(data["Time"], data["TOT"], label='TOT')
    plt.xlabel("Time (s)")
    plt.ylabel("Power Consumption (mW)")
    plt.savefig(os.path.join(os.getcwd(),'Results',result_folder,'Trt_Jetson_CV_Consumption_rep_1000.png'))

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

def audio_tagging_rep(output_data, nb_rep):

	global start_event_main
	start_event_main.record()
	global start_audio_tag
	start_audio_tag = datetime.now()

    # the spectrogram is loaded
	input_data = np.load(os.path.abspath(os.path.join(os.getcwd(),args.audio_path)))
	input_data = input_data.reshape(1,1,1,96000)
	

	# Load trt engine file
	model_path_trt = os.path.join(os.getcwd(),args.engine_path)
	print(os.getcwd())
	
	with open(model_path_trt, 'rb') as f:
		engine_data = f.read()
	runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
	engine = runtime.deserialize_cuda_engine(engine_data)
	context = engine.create_execution_context()

	# Size Output :
	output_shape = output_data.shape

	# GPU Warm Up
	global start_warmUp
	start_warmUp=(datetime.now() - start_audio_tag).total_seconds()
	dummy_input = np.random.rand(1,224000)
	dummy_output = np.empty(output_shape, dtype=np.float32)
	dummy_input_buf = cuda.mem_alloc(dummy_input.nbytes)
	cuda.memcpy_htod_async(dummy_input_buf, dummy_input)
	dummy_output_buf = cuda.mem_alloc(output_shape[0] * np.dtype(np.float32).itemsize)
	context.execute_v2(bindings=[int(dummy_input_buf),int(dummy_output_buf)])
	cuda.memcpy_dtoh_async(dummy_output, dummy_output_buf)
	global end_warmpUp
	end_warmpUp=(datetime.now() - start_audio_tag).total_seconds()

	# Allocate input and output buffers on the GPU :
	input_buf = cuda.mem_alloc(input_data.nbytes)
	output_buf = cuda.mem_alloc(output_shape[0] * np.dtype(np.float32).itemsize)

	# Transfer input data to GPU
	cuda.memcpy_htod_async(input_buf, input_data)

	# Inference Time Variables & Start Recording :
	global start_value
	global inference_start
	global start_time_infer
	start_infer=cuda.Event()
	end_event=cuda.Event()

	inference_start= (datetime.now() - start_value).total_seconds()

	global inferences_time_list

	start_infer.record()
	start_time_infer=start_event_main.time_till(start_infer)/1000
	inference_start= (datetime.now() - start_value).total_seconds()
	inferences_time_list.append(inference_start)

	start_one_infer=cuda.Event()
	end_one_event=cuda.Event()

	global inference_duration
	inference_duration=[]
	for i in range(nb_rep):
	    start_one_infer.record()
	    context.execute_v2(bindings=[int(input_buf),int(output_buf)])
	    end_one_event.record()
	    end_one_event.synchronize()
	    elapsed_one_time=start_one_infer.time_till(end_one_event)/1000
	    inference_duration.append(elapsed_one_time)

	end_event.record()
	end_event.synchronize()
	elapsed_time=start_infer.time_till(end_event)/1000
	inference_end= (datetime.now() - start_value).total_seconds()
	inferences_time_list.append(inference_end)
        # Transfer output data from GPU :
	cuda.memcpy_dtoh_async(output_data, output_buf)

	# Stop Consumption Recording :
	global thread_stop
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
    parser.add_argument('--classes_name', required=True)


    args = parser.parse_args()
    nb_rep=int(args.nb_rep)

    global result_folder
    result_folder = str(datetime.now().strftime("%Y-%m-%d %H:%M")) + f" {args.engine}" + f" {args.power_mode}"

    os.mkdir(os.path.join(os.getcwd(),'Results',result_folder))

    # Creation of the output
    #classes_name = np.load(os.path.join(os.getcwd(),'Sources','classes.npy'))
    classes_name=np.load(os.path.join(os.path.dirname(os.getcwd()),args.classes_name))
    output_shape = (len(classes_name)-1,)
    output_data = np.empty(output_shape, dtype=np.float32)

    end_event_main=cuda.Event()
    thread_conso=Thread(target = power_conso)
    thread_conso.start()
    thread_model=Thread(target = audio_tagging_rep, args=(output_data,nb_rep))
    end_event_main.record()

    # Print Result

    audio_tagging_rep(output_data,nb_rep)
    

    framewise_output = output_data
    sorted_indexes = np.argsort(framewise_output)[::-1]

    top_k = 10  # Show top results
    top_result_mat = framewise_output[sorted_indexes[0 : top_k]]
    top_classes = classes_name[sorted_indexes[0 : top_k]]
    for i in range(top_k):
        print("* ",i+1, "result : ", top_result_mat[i], " with class : ",top_classes[i],"\n")

    # Store parameters
    with open(os.path.join(os.getcwd(),'Results',result_folder,'parameters'),'w') as param:
        param.write("Engine : ")
        param.write(args.engine)
        param.write("\n")
        param.write("Power Mode : ")
        param.write(args.power_mode)

    time.sleep(20)

