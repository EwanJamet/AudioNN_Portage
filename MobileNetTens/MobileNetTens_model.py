import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.onnx
import numpy as np

def init_layer(layer):
    """Initialize a Linear or Convolutional layer. """
    nn.init.xavier_uniform_(layer.weight)
 
    if hasattr(layer, 'bias'):
        if layer.bias is not None:
            layer.bias.data.fill_(0.)
            
    
def init_bn(bn):
    """Initialize a Batchnorm layer. """
    bn.bias.data.fill_(0.)
    bn.weight.data.fill_(1.)


class MobileNetTens(nn.Module):
    def __init__(self, classes_num):
        
        super(MobileNetTens, self).__init__()

        self.bn0 = nn.BatchNorm2d(64)

        def conv_bn(inp, oup, stride):
            _layers = [
                nn.Conv2d(inp, oup, 3, 1, 1, bias=False), 
                nn.AvgPool2d(stride), 
                nn.BatchNorm2d(oup), 
                nn.ReLU(inplace=True)
                ]
            _layers = nn.Sequential(*_layers)
            init_layer(_layers[0])
            init_bn(_layers[2])
            return _layers

        def conv_dw(inp, oup, stride):
            _layers = [
                nn.Conv2d(inp, inp, 3, 1, 1, groups=inp, bias=False), 
                nn.AvgPool2d(stride), 
                nn.BatchNorm2d(inp), 
                nn.ReLU(inplace=True), 
                nn.Conv2d(inp, oup, 1, 1, 0, bias=False), 
                nn.BatchNorm2d(oup), 
                nn.ReLU(inplace=True)
                ]
            _layers = nn.Sequential(*_layers)
            init_layer(_layers[0])
            init_bn(_layers[2])
            init_layer(_layers[4])
            init_bn(_layers[5])
            return _layers

        self.features = nn.Sequential(
            conv_bn(  1,  32, 2), 
            conv_dw( 32,  64, 1),
            conv_dw( 64, 128, 2),
            conv_dw(128, 128, 1),
            conv_dw(128, 256, 2),
            conv_dw(256, 256, 1),
            conv_dw(256, 512, 2),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 1024, 2),
            conv_dw(1024, 1024, 1))

        self.fc1 = nn.Linear(1024, 1024, bias=True)
        self.fc_audioset = nn.Linear(1024, classes_num, bias=True)

        self.init_weights()

    def init_weights(self):
        init_bn(self.bn0)
        init_layer(self.fc1)
        init_layer(self.fc_audioset)
 
    def forward(self, input, mixup_lambda=None):
        """
        Input: (batch_size, data_length)"""
        x = input
        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)
        
        x = self.features(x)
        x = torch.mean(x, dim=3)
        
        (x1, _) = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2
        x = F.relu_(self.fc1(x))
        clipwise_output = torch.sigmoid(self.fc_audioset(x))
        
        output_dict = {'clipwise_output': clipwise_output}

        return output_dict


def Convert_ONNX():

    # torch_model = MobileNetV1(sample_rate, window_size, hop_size, mel_bins, fmin, fmax, classes_num)
    res_height = 701 # need to put the good dimensions (need to be fix but i want it to adapt for the right size)
    res_width = 64
    path_model = "/home/ewan/Documents/Ecole/Procom/audioset_tagging_cnn/MobileNetTens/MobileNetTens_W.onnx"
    torch.onnx.export(
                model,# your pytorch model
                torch.randn(1,1,res_height, res_width, device="cpu"), # need to change to put the right entrance
                path_model, # path to save
                export_params=True,
                opset_version=10, # mandatory for tensil
                input_names = ['Input'],
                output_names=["Output"],
    )

classes_num = 527
model = MobileNetTens(classes_num)
path = "/home/ewan/Documents/Ecole/Procom/audioset_tagging_cnn/MobileNetV1.pth"
checkpoint = torch.load(path, map_location=torch.device('cpu'))

key_before = np.array(['spectrogram_extractor.stft.conv_real.weight', 'spectrogram_extractor.stft.conv_imag.weight', 'logmel_extractor.melW', 'bn0.weight', 'bn0.bias', 'bn0.running_mean', 'bn0.running_var', 'bn0.num_batches_tracked', 'features.0.0.weight', 'features.0.2.weight', 'features.0.2.bias', 'features.0.2.running_mean', 'features.0.2.running_var', 'features.0.2.num_batches_tracked', 'features.1.0.weight', 'features.1.2.weight', 'features.1.2.bias', 'features.1.2.running_mean', 'features.1.2.running_var', 'features.1.2.num_batches_tracked', 'features.1.4.weight', 'features.1.5.weight', 'features.1.5.bias', 'features.1.5.running_mean', 'features.1.5.running_var', 'features.1.5.num_batches_tracked', 'features.2.0.weight', 'features.2.2.weight', 'features.2.2.bias', 'features.2.2.running_mean', 'features.2.2.running_var', 'features.2.2.num_batches_tracked', 'features.2.4.weight', 'features.2.5.weight', 'features.2.5.bias', 'features.2.5.running_mean', 'features.2.5.running_var', 'features.2.5.num_batches_tracked', 'features.3.0.weight', 'features.3.2.weight', 'features.3.2.bias', 'features.3.2.running_mean', 'features.3.2.running_var', 'features.3.2.num_batches_tracked', 'features.3.4.weight', 'features.3.5.weight', 'features.3.5.bias', 'features.3.5.running_mean', 'features.3.5.running_var', 'features.3.5.num_batches_tracked', 'features.4.0.weight', 'features.4.2.weight', 'features.4.2.bias', 'features.4.2.running_mean', 'features.4.2.running_var', 'features.4.2.num_batches_tracked', 'features.4.4.weight', 'features.4.5.weight', 'features.4.5.bias', 'features.4.5.running_mean', 'features.4.5.running_var', 'features.4.5.num_batches_tracked', 'features.5.0.weight', 'features.5.2.weight', 'features.5.2.bias', 'features.5.2.running_mean', 'features.5.2.running_var', 'features.5.2.num_batches_tracked', 'features.5.4.weight', 'features.5.5.weight', 'features.5.5.bias', 'features.5.5.running_mean', 'features.5.5.running_var', 'features.5.5.num_batches_tracked', 'features.6.0.weight', 'features.6.2.weight', 'features.6.2.bias', 'features.6.2.running_mean', 'features.6.2.running_var', 'features.6.2.num_batches_tracked', 'features.6.4.weight', 'features.6.5.weight', 'features.6.5.bias', 'features.6.5.running_mean', 'features.6.5.running_var', 'features.6.5.num_batches_tracked', 'features.7.0.weight', 'features.7.2.weight', 'features.7.2.bias', 'features.7.2.running_mean', 'features.7.2.running_var', 'features.7.2.num_batches_tracked', 'features.7.4.weight', 'features.7.5.weight', 'features.7.5.bias', 'features.7.5.running_mean', 'features.7.5.running_var', 'features.7.5.num_batches_tracked', 'features.8.0.weight', 'features.8.2.weight', 'features.8.2.bias', 'features.8.2.running_mean', 'features.8.2.running_var', 'features.8.2.num_batches_tracked', 'features.8.4.weight', 'features.8.5.weight', 'features.8.5.bias', 'features.8.5.running_mean', 'features.8.5.running_var', 'features.8.5.num_batches_tracked', 'features.9.0.weight', 'features.9.2.weight', 'features.9.2.bias', 'features.9.2.running_mean', 'features.9.2.running_var', 'features.9.2.num_batches_tracked', 'features.9.4.weight', 'features.9.5.weight', 'features.9.5.bias', 'features.9.5.running_mean', 'features.9.5.running_var', 'features.9.5.num_batches_tracked', 'features.10.0.weight', 'features.10.2.weight', 'features.10.2.bias', 'features.10.2.running_mean', 'features.10.2.running_var', 'features.10.2.num_batches_tracked', 'features.10.4.weight', 'features.10.5.weight', 'features.10.5.bias', 'features.10.5.running_mean', 'features.10.5.running_var', 'features.10.5.num_batches_tracked', 'features.11.0.weight', 'features.11.2.weight', 'features.11.2.bias', 'features.11.2.running_mean', 'features.11.2.running_var', 'features.11.2.num_batches_tracked', 'features.11.4.weight', 'features.11.5.weight', 'features.11.5.bias', 'features.11.5.running_mean', 'features.11.5.running_var', 'features.11.5.num_batches_tracked', 'features.12.0.weight', 'features.12.2.weight', 'features.12.2.bias', 'features.12.2.running_mean', 'features.12.2.running_var', 'features.12.2.num_batches_tracked', 'features.12.4.weight', 'features.12.5.weight', 'features.12.5.bias', 'features.12.5.running_mean', 'features.12.5.running_var', 'features.12.5.num_batches_tracked', 'features.13.0.weight', 'features.13.2.weight', 'features.13.2.bias', 'features.13.2.running_mean', 'features.13.2.running_var', 'features.13.2.num_batches_tracked', 'features.13.4.weight', 'features.13.5.weight', 'features.13.5.bias', 'features.13.5.running_mean', 'features.13.5.running_var', 'features.13.5.num_batches_tracked', 'fc1.weight', 'fc1.bias', 'fc_audioset.weight', 'fc_audioset.bias'])
key_needed = np.array(["bn0.weight", "bn0.bias", "bn0.running_mean", "bn0.running_var", "features.0.0.weight", "features.0.2.weight", "features.0.2.bias", "features.0.2.running_mean", "features.0.2.running_var", "features.1.0.weight", "features.1.2.weight", "features.1.2.bias", "features.1.2.running_mean", "features.1.2.running_var", "features.1.4.weight", "features.1.5.weight", "features.1.5.bias", "features.1.5.running_mean", "features.1.5.running_var", "features.2.0.weight", "features.2.2.weight", "features.2.2.bias", "features.2.2.running_mean", "features.2.2.running_var", "features.2.4.weight", "features.2.5.weight", "features.2.5.bias", "features.2.5.running_mean", "features.2.5.running_var", "features.3.0.weight", "features.3.2.weight", "features.3.2.bias", "features.3.2.running_mean", "features.3.2.running_var", "features.3.4.weight", "features.3.5.weight", "features.3.5.bias", "features.3.5.running_mean", "features.3.5.running_var", "features.4.0.weight", "features.4.2.weight", "features.4.2.bias", "features.4.2.running_mean", "features.4.2.running_var", "features.4.4.weight", "features.4.5.weight", "features.4.5.bias", "features.4.5.running_mean", "features.4.5.running_var", "features.5.0.weight", "features.5.2.weight", "features.5.2.bias", "features.5.2.running_mean", "features.5.2.running_var", "features.5.4.weight", "features.5.5.weight", "features.5.5.bias", "features.5.5.running_mean", "features.5.5.running_var", "features.6.0.weight", "features.6.2.weight", "features.6.2.bias", "features.6.2.running_mean", "features.6.2.running_var", "features.6.4.weight", "features.6.5.weight", "features.6.5.bias", "features.6.5.running_mean", "features.6.5.running_var", "features.7.0.weight", "features.7.2.weight", "features.7.2.bias", "features.7.2.running_mean", "features.7.2.running_var", "features.7.4.weight", "features.7.5.weight", "features.7.5.bias", "features.7.5.running_mean", "features.7.5.running_var", "features.8.0.weight", "features.8.2.weight", "features.8.2.bias", "features.8.2.running_mean", "features.8.2.running_var", "features.8.4.weight", "features.8.5.weight", "features.8.5.bias", "features.8.5.running_mean", "features.8.5.running_var", "features.9.0.weight", "features.9.2.weight", "features.9.2.bias", "features.9.2.running_mean", "features.9.2.running_var", "features.9.4.weight", "features.9.5.weight", "features.9.5.bias", "features.9.5.running_mean", "features.9.5.running_var", "features.10.0.weight", "features.10.2.weight", "features.10.2.bias", "features.10.2.running_mean", "features.10.2.running_var", "features.10.4.weight", "features.10.5.weight", "features.10.5.bias", "features.10.5.running_mean", "features.10.5.running_var", "features.11.0.weight", "features.11.2.weight", "features.11.2.bias", "features.11.2.running_mean", "features.11.2.running_var", "features.11.4.weight", "features.11.5.weight", "features.11.5.bias", "features.11.5.running_mean", "features.11.5.running_var", "features.12.0.weight", "features.12.2.weight", "features.12.2.bias", "features.12.2.running_mean", "features.12.2.running_var", "features.12.4.weight", "features.12.5.weight", "features.12.5.bias", "features.12.5.running_mean", "features.12.5.running_var", "features.13.0.weight", "features.13.2.weight", "features.13.2.bias", "features.13.2.running_mean", "features.13.2.running_var", "features.13.4.weight", "features.13.5.weight", "features.13.5.bias", "features.13.5.running_mean", "features.13.5.running_var", "fc1.weight", "fc1.bias", "fc_audioset.weight", "fc_audioset.bias"])

key_to_suppress = np.setdiff1d(key_before, key_needed)

key_not_to_suppress = np.array(["bn0.num_batches_tracked", "features.0.2.num_batches_tracked", "features.1.2.num_batches_tracked", "features.1.5.num_batches_tracked", "features.2.2.num_batches_tracked", "features.2.5.num_batches_tracked", "features.3.2.num_batches_tracked", "features.3.5.num_batches_tracked", "features.4.2.num_batches_tracked", "features.4.5.num_batches_tracked", "features.5.2.num_batches_tracked", "features.5.5.num_batches_tracked", "features.6.2.num_batches_tracked", "features.6.5.num_batches_tracked", "features.7.2.num_batches_tracked", "features.7.5.num_batches_tracked", "features.8.2.num_batches_tracked", "features.8.5.num_batches_tracked", "features.9.2.num_batches_tracked", "features.9.5.num_batches_tracked", "features.10.2.num_batches_tracked", "features.10.5.num_batches_tracked", "features.11.2.num_batches_tracked", "features.11.5.num_batches_tracked", "features.12.2.num_batches_tracked", "features.12.5.num_batches_tracked", "features.13.2.num_batches_tracked", "features.13.5.num_batches_tracked"])

key_to_suppress = np.setdiff1d(key_to_suppress,key_not_to_suppress)

for key in key_to_suppress:
    del checkpoint['model'][key]

key_add_suppress = np.array(["bn0.num_batches_tracked", "features.0.2.num_batches_tracked", "features.1.2.num_batches_tracked", "features.1.5.num_batches_tracked", "features.2.2.num_batches_tracked", "features.2.5.num_batches_tracked", "features.3.2.num_batches_tracked", "features.3.5.num_batches_tracked", "features.4.2.num_batches_tracked", "features.4.5.num_batches_tracked", "features.5.2.num_batches_tracked", "features.5.5.num_batches_tracked", "features.6.2.num_batches_tracked", "features.6.5.num_batches_tracked", "features.7.2.num_batches_tracked", "features.7.5.num_batches_tracked", "features.8.2.num_batches_tracked", "features.8.5.num_batches_tracked", "features.9.2.num_batches_tracked", "features.9.5.num_batches_tracked", "features.10.2.num_batches_tracked", "features.10.5.num_batches_tracked", "features.11.2.num_batches_tracked", "features.11.5.num_batches_tracked", "features.12.2.num_batches_tracked", "features.12.5.num_batches_tracked", "features.13.2.num_batches_tracked", "features.13.5.num_batches_tracked"])

model.load_state_dict(checkpoint['model'])

Convert_ONNX()
print("conversion complete")
