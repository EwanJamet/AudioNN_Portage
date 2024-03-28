import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '../../utils'))
import numpy as np
import argparse
import librosa
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import torch

from utilities import create_folder, get_filename
from models import *
from pytorch_utils import move_data_to_device
import config
import time
from threading import Thread
from jtop import jtop
import time
import  csv
from datetime import datetime, timedelta



thread_stop=False
start_value=datetime.now()
inference_start=datetime.now()
start_event_main=torch.cuda.Event(enable_timing=True)
start_time_infer=datetime.now()

def power_conso():
    print(os.path.join(os.getcwd(),'Results','P.png'))
    global start_event_main
    start_event_main.record()
    with jtop() as jetson:
    # jetson.ok() will provide the proper update frequency
        global thread_stop
        data={"Total Power" : [], "GPU" : [], "Time":[]}
        while jetson.ok() and thread_stop==False:
            # Read tegra stats
            data["Total Power"].append(jetson.stats['Power TOT'])
            data["GPU"].append(jetson.stats['GPU'])
            data["Time"].append(jetson.stats['time'])
    global start_value
    global start_time_infer
    start_value = data["Time"][0]
    data["Time"] = [(time_value - start_value).total_seconds() for time_value in data["Time"] ]
    
    plt.plot(data["Time"], data["GPU"])
    plt.xlabel("Time (s)")
    plt.ylabel("GPU Consumption (mW)")
    plt.axvline(x=start_time_infer, color='red')
    plt.savefig(os.path.join(os.getcwd(),'Results','Pytorch_GPU_Power_Consumption_avec_sync.png'))
    
    with open(os.path.join(os.getcwd(),'Results','Power_Consumption_bis.csv'), 'w', newline='') as f:
    	writer = csv.writer(f)
    	writer.writerow(data.keys())
    	writer.writerows(zip(*data.values()))
    
 
    
    	           
def audio_tagging(args):
    """Inference audio tagging result of an audio clip.
    """
    
    test =torch.cuda.Event(enable_timing=True)
    test.record()
    global start_event_main
    time.sleep(10)
    
    # Arugments & parameters
    sample_rate = args.sample_rate
    window_size = args.window_size
    hop_size = args.hop_size
    mel_bins = args.mel_bins
    fmin = args.fmin
    fmax = args.fmax
    model_type = args.model_type
    checkpoint_path = args.checkpoint_path
    audio_path = args.audio_path
    device = torch.device('cuda') if args.cuda and torch.cuda.is_available() else torch.device('cpu')
    
    classes_num = config.classes_num
    labels = config.labels

    # Model
    Model = eval(model_type)
    model = Model(sample_rate=sample_rate, window_size=window_size, 
        hop_size=hop_size, mel_bins=mel_bins, fmin=fmin, fmax=fmax, 
        classes_num=classes_num)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model'])

    # Parallel
    if 'cuda' in str(device):
        model.to(device)
        print('GPU number: {}'.format(torch.cuda.device_count()))
        model = torch.nn.DataParallel(model)
    else:
        print('Using CPU.')
    # Load audio
    (waveform, _) = librosa.core.load(audio_path, sr=sample_rate, mono=True)

    waveform = waveform[None, :]    # (1, audio_length)
    waveform = move_data_to_device(waveform, device)

    # INIT LOGGERS
    #starter, ender = torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
    repetitions = 1
    timings=np.zeros((repetitions,1))
    global start_value	
    global inference_start
    global start_time_infer
    	
    # dummy input for warm-up
    dummy_input = torch.randn(1,224000, dtype=torch.float).to(device)
    
    # GPU warm-up
    #for _ in range(10):
    #    _ = model(dummy_input)
    _ = model(dummy_input) 
        					
    # Forward
    
    with torch.no_grad():

        starter=torch.cuda.Event(enable_timing=True)
        ender=torch.cuda.Event(enable_timing=True)
        starter.record()
        model.eval()
        
        ender.record()
        torch.cuda.synchronize()
        #ender.synchronize()
        inference_start= (datetime.now() - start_value).total_seconds()
        start_time_infer=start_event_main.elapsed_time(starter)/1000
        print("start_time_infer with elasped_time (s) : ",start_time_infer)
        print("start_time_infer with datetime (s) : ",inference_start)
        batch_output_dict = model(waveform, None)
	# WAIT FOR GPU SYNC
        #torch.cuda.synchronize()
        curr_time = starter.elapsed_time(ender)
        timings[0] = curr_time
        print('inference time:', timings[0])

       

    clipwise_output = batch_output_dict['clipwise_output'].data.cpu().numpy()[0]
    """(classes_num,)"""

    sorted_indexes = np.argsort(clipwise_output)[::-1]

    # Print audio tagging top probabilities
    for k in range(10):
        print('{}: {:.3f}'.format(np.array(labels)[sorted_indexes[k]], 
            clipwise_output[sorted_indexes[k]]))

    # Print embedding
    if 'embedding' in batch_output_dict.keys():
        embedding = batch_output_dict['embedding'].data.cpu().numpy()[0]
        print('embedding: {}'.format(embedding.	shape))
    global thread_stop
    time.sleep(20)
    thread_stop=True
    
    return clipwise_output, labels


def sound_event_detection(args):
    """Inference sound event detection result of an audio clip.
    """

    # Arugments & parameters
    sample_rate = args.sample_rate
    window_size = args.window_size
    hop_size = args.hop_size
    mel_bins = args.mel_bins
    fmin = args.fmin
    fmax = args.fmax
    model_type = args.model_type
    checkpoint_path = args.checkpoint_path
    audio_path = args.audio_path
    device = torch.device('cuda') if args.cuda and torch.cuda.is_available() else torch.device('cpu')

    classes_num = config.classes_num
    labels = config.labels
    frames_per_second = sample_rate // hop_size

    # Paths
    fig_path = os.path.join('results', '{}.png'.format(get_filename(audio_path)))
    create_folder(os.path.dirname(fig_path))

    # Model
    Model = eval(model_type)
    model = Model(sample_rate=sample_rate, window_size=window_size, 
        hop_size=hop_size, mel_bins=mel_bins, fmin=fmin, fmax=fmax, 
        classes_num=classes_num)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model'])

    # Parallel
    print('GPU number: {}'.format(torch.cuda.device_count()))
    model = torch.nn.DataParallel(model)

    if 'cuda' in str(device):
        model.to(device)
    
    # Load audio
    (waveform, _) = librosa.core.load(audio_path, sr=sample_rate, mono=True)

    waveform = waveform[None, :]    # (1, audio_length)
    waveform = move_data_to_device(waveform, device)
    
    
        
    # Forward
    with torch.no_grad():
        model.eval()
        batch_output_dict = model(waveform, None)

    framewise_output = batch_output_dict['framewise_output'].data.cpu().numpy()[0]
    """(time_steps, classes_num)"""

    print('Sound event detection result (time_steps x classes_num): {}'.format(
        framewise_output.shape))

    sorted_indexes = np.argsort(np.max(framewise_output, axis=0))[::-1]

    top_k = 10  # Show top results
    top_result_mat = framewise_output[:, sorted_indexes[0 : top_k]]    
    """(time_steps, top_k)"""

    # Plot result    
    stft = librosa.core.stft(y=waveform[0].data.cpu().numpy(), n_fft=window_size, 
        hop_length=hop_size, window='hann', center=True)
    frames_num = stft.shape[-1]

    fig, axs = plt.subplots(2, 1, sharex=True, figsize=(10, 4))
    axs[0].matshow(np.log(np.abs(stft)), origin='lower', aspect='auto', cmap='jet')
    axs[0].set_ylabel('Frequency bins')
    axs[0].set_title('Log spectrogram')
    axs[1].matshow(top_result_mat.T, origin='upper', aspect='auto', cmap='jet', vmin=0, vmax=1)
    axs[1].xaxis.set_ticks(np.arange(0, frames_num, frames_per_second))
    axs[1].xaxis.set_ticklabels(np.arange(0, frames_num / frames_per_second))
    axs[1].yaxis.set_ticks(np.arange(0, top_k))
    axs[1].yaxis.set_ticklabels(np.array(labels)[sorted_indexes[0 : top_k]])
    axs[1].yaxis.grid(color='k', linestyle='solid', linewidth=0.3, alpha=0.3)
    axs[1].set_xlabel('Seconds')
    axs[1].xaxis.set_ticks_position('bottom')

    plt.tight_layout()
    plt.savefig(fig_path)
    print('Save sound event detection visualization to {}'.format(fig_path))
    return framewise_output, labels


if __name__ == '__main__':
    #print(os.path.abspath(os.path.join(os.path.dirname(__file__),'..','Results')))
    #print(os.getcwd())
    parser = argparse.ArgumentParser(description='Example of parser. ')
    subparsers = parser.add_subparsers(dest='mode')

    parser_at = subparsers.add_parser('audio_tagging')
    parser_at.add_argument('--sample_rate', type=int, default=32000)
    parser_at.add_argument('--window_size', type=int, default=1024)
    parser_at.add_argument('--hop_size', type=int, default=320)
    parser_at.add_argument('--mel_bins', type=int, default=64)
    parser_at.add_argument('--fmin', type=int, default=50)
    parser_at.add_argument('--fmax', type=int, default=14000) 
    parser_at.add_argument('--model_type', type=str, required=True)
    parser_at.add_argument('--checkpoint_path', type=str, required=True)
    parser_at.add_argument('--audio_path', type=str, required=True)
    parser_at.add_argument('--cuda', action='store_true', default=False)

    parser_sed = subparsers.add_parser('sound_event_detection')
    parser_sed.add_argument('--sample_rate', type=int, default=32000)
    parser_sed.add_argument('--window_size', type=int, default=1024)
    parser_sed.add_argument('--hop_size', type=int, default=320)
    parser_sed.add_argument('--mel_bins', type=int, default=64)
    parser_sed.add_argument('--fmin', type=int, default=50)
    parser_sed.add_argument('--fmax', type=int, default=14000) 
    parser_sed.add_argument('--model_type', type=str, required=True)
    parser_sed.add_argument('--checkpoint_path', type=str, required=True)
    parser_sed.add_argument('--audio_path', type=str, required=True)
    parser_sed.add_argument('--cuda', action='store_true', default=False)
    
    
    args = parser.parse_args()
    print(args.audio_path)
    
    if  args.mode == 'audio_tagging':

        thread_conso=Thread(target = power_conso)
        thread_conso.start()
        thread_model=Thread(target = audio_tagging, args=(args,))
        thread_model.start()

    elif args.mode == 'sound_event_detection':
        sound_event_detection(args)

    else:
        raise Exception('Error argument!')
        
        

