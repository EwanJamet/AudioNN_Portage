import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.onnx
import numpy as np
from torchinfo import summary
from onnxsim import simplify
import onnx as onnx

print()

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


class LeeNetConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        
        super(LeeNetConvBlock, self).__init__()
        
        self.conv1 =  nn.Conv2d(in_channels=in_channels, 
                              out_channels=out_channels,
                              kernel_size=(1, kernel_size), stride=(1, stride),
                              padding=(1, kernel_size // 2) , bias=False)

                              
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.init_weight()
        
    def init_weight(self):
        init_layer(self.conv1)
        init_bn(self.bn1)

    def forward(self, x, pool_size=1):
        x = F.relu_(self.bn1(self.conv1(x)))
        if pool_size != 1:
            x = F.max_pool2d(x, kernel_size=pool_size, padding=pool_size // 2)
        return x


class LeeNet11(nn.Module):
    def __init__(self, classes_num):
        
        super(LeeNet11, self).__init__()

        window = 'hann'
        center = True
        pad_mode = 'reflect'
        ref = 1.0
        amin = 1e-10
        top_db = None

        self.conv_block1 = LeeNetConvBlock(1, 64, 3, 3)
        self.conv_block2 = LeeNetConvBlock(64, 64, 3, 1)
        self.conv_block3 = LeeNetConvBlock(64, 64, 3, 1)
        self.conv_block4 = LeeNetConvBlock(64, 128, 3, 1)
        self.conv_block5 = LeeNetConvBlock(128, 128, 3, 1)
        self.conv_block6 = LeeNetConvBlock(128, 128, 3, 1)
        self.conv_block7 = LeeNetConvBlock(128, 128, 3, 1)
        self.conv_block8 = LeeNetConvBlock(128, 128, 3, 1)
        self.conv_block9 = LeeNetConvBlock(128, 256, 3, 1)
        

        self.fc1 = nn.Linear(256, 512, bias=True)
        self.fc_audioset = nn.Linear(512, classes_num, bias=True)
        
        self.init_weight()

    def init_weight(self):
        init_layer(self.fc1)
        init_layer(self.fc_audioset)
 
    def forward(self, input):
        """
        Input: (batch_size, data_length)"""
        # print(input.shape)
        # x = torch.squeeze(input,dim=1)
        # x = torch.unsqueeze(input, dim=0)

        # print(x.shape)
        x = self.conv_block1(input)
        x = self.conv_block2(x, pool_size=3)
        x = self.conv_block3(x, pool_size=3)
        x = self.conv_block4(x, pool_size=3)
        x = self.conv_block5(x, pool_size=3)
        x = self.conv_block6(x, pool_size=3)
        x = self.conv_block7(x, pool_size=3)
        x = self.conv_block8(x, pool_size=3)
        x = self.conv_block9(x, pool_size=3)
        
        # (x1, _) = torch.max(x, dim=2)
        # x2 = torch.mean(x, dim=2)
        # x = x1 + x2
        # x = F.relu_(self.fc1(x))
        # clipwise_output = torch.sigmoid(self.fc_audioset(x))
        
        # output_dict = {'clipwise_output': clipwise_output}
        output_dict = x
        return output_dict





def Convert_ONNX(path_model):

    # torch_model = MobileNetV1(sample_rate, window_size, hop_size, mel_bins, fmin, fmax, classes_num)
    res_height = 1 # need to put the good dimensions (need to be fix but i want it to adapt for the right size)
    res_width = 50000
    
    x = torch.randn(1,1,res_height, res_width, device="cpu")
    print("input : ",x.shape)
    torch.onnx.export(
                model,# your pytorch model
                x, # need to change to put the right entrance
                path_model, # path to save
                export_params=True,
                opset_version=10, # mandatory for tensil 10 by default
                input_names=["input.1"],
                output_names=["Output"],
    )

classes_num = 527
model = LeeNet11(classes_num)
path = "../audioset_tagging_cnn/MobileNetTens/LeeNet11.pth"
checkpoint = torch.load(path, map_location=torch.device('cpu'))

key_before = np.array(['spectrogram_extractor.stft.conv_real.weight', 'spectrogram_extractor.stft.conv_imag.weight', 'logmel_extractor.melW', 'bn0.weight', 'bn0.bias', 'bn0.running_mean', 'bn0.running_var', 'bn0.num_batches_tracked', 'features.0.0.weight', 'features.0.2.weight', 'features.0.2.bias', 'features.0.2.running_mean', 'features.0.2.running_var', 'features.0.2.num_batches_tracked', 'features.1.0.weight', 'features.1.2.weight', 'features.1.2.bias', 'features.1.2.running_mean', 'features.1.2.running_var', 'features.1.2.num_batches_tracked', 'features.1.4.weight', 'features.1.5.weight', 'features.1.5.bias', 'features.1.5.running_mean', 'features.1.5.running_var', 'features.1.5.num_batches_tracked', 'features.2.0.weight', 'features.2.2.weight', 'features.2.2.bias', 'features.2.2.running_mean', 'features.2.2.running_var', 'features.2.2.num_batches_tracked', 'features.2.4.weight', 'features.2.5.weight', 'features.2.5.bias', 'features.2.5.running_mean', 'features.2.5.running_var', 'features.2.5.num_batches_tracked', 'features.3.0.weight', 'features.3.2.weight', 'features.3.2.bias', 'features.3.2.running_mean', 'features.3.2.running_var', 'features.3.2.num_batches_tracked', 'features.3.4.weight', 'features.3.5.weight', 'features.3.5.bias', 'features.3.5.running_mean', 'features.3.5.running_var', 'features.3.5.num_batches_tracked', 'features.4.0.weight', 'features.4.2.weight', 'features.4.2.bias', 'features.4.2.running_mean', 'features.4.2.running_var', 'features.4.2.num_batches_tracked', 'features.4.4.weight', 'features.4.5.weight', 'features.4.5.bias', 'features.4.5.running_mean', 'features.4.5.running_var', 'features.4.5.num_batches_tracked', 'features.5.0.weight', 'features.5.2.weight', 'features.5.2.bias', 'features.5.2.running_mean', 'features.5.2.running_var', 'features.5.2.num_batches_tracked', 'features.5.4.weight', 'features.5.5.weight', 'features.5.5.bias', 'features.5.5.running_mean', 'features.5.5.running_var', 'features.5.5.num_batches_tracked', 'features.6.0.weight', 'features.6.2.weight', 'features.6.2.bias', 'features.6.2.running_mean', 'features.6.2.running_var', 'features.6.2.num_batches_tracked', 'features.6.4.weight', 'features.6.5.weight', 'features.6.5.bias', 'features.6.5.running_mean', 'features.6.5.running_var', 'features.6.5.num_batches_tracked', 'features.7.0.weight', 'features.7.2.weight', 'features.7.2.bias', 'features.7.2.running_mean', 'features.7.2.running_var', 'features.7.2.num_batches_tracked', 'features.7.4.weight', 'features.7.5.weight', 'features.7.5.bias', 'features.7.5.running_mean', 'features.7.5.running_var', 'features.7.5.num_batches_tracked', 'features.8.0.weight', 'features.8.2.weight', 'features.8.2.bias', 'features.8.2.running_mean', 'features.8.2.running_var', 'features.8.2.num_batches_tracked', 'features.8.4.weight', 'features.8.5.weight', 'features.8.5.bias', 'features.8.5.running_mean', 'features.8.5.running_var', 'features.8.5.num_batches_tracked', 'features.9.0.weight', 'features.9.2.weight', 'features.9.2.bias', 'features.9.2.running_mean', 'features.9.2.running_var', 'features.9.2.num_batches_tracked', 'features.9.4.weight', 'features.9.5.weight', 'features.9.5.bias', 'features.9.5.running_mean', 'features.9.5.running_var', 'features.9.5.num_batches_tracked', 'features.10.0.weight', 'features.10.2.weight', 'features.10.2.bias', 'features.10.2.running_mean', 'features.10.2.running_var', 'features.10.2.num_batches_tracked', 'features.10.4.weight', 'features.10.5.weight', 'features.10.5.bias', 'features.10.5.running_mean', 'features.10.5.running_var', 'features.10.5.num_batches_tracked', 'features.11.0.weight', 'features.11.2.weight', 'features.11.2.bias', 'features.11.2.running_mean', 'features.11.2.running_var', 'features.11.2.num_batches_tracked', 'features.11.4.weight', 'features.11.5.weight', 'features.11.5.bias', 'features.11.5.running_mean', 'features.11.5.running_var', 'features.11.5.num_batches_tracked', 'features.12.0.weight', 'features.12.2.weight', 'features.12.2.bias', 'features.12.2.running_mean', 'features.12.2.running_var', 'features.12.2.num_batches_tracked', 'features.12.4.weight', 'features.12.5.weight', 'features.12.5.bias', 'features.12.5.running_mean', 'features.12.5.running_var', 'features.12.5.num_batches_tracked', 'features.13.0.weight', 'features.13.2.weight', 'features.13.2.bias', 'features.13.2.running_mean', 'features.13.2.running_var', 'features.13.2.num_batches_tracked', 'features.13.4.weight', 'features.13.5.weight', 'features.13.5.bias', 'features.13.5.running_mean', 'features.13.5.running_var', 'features.13.5.num_batches_tracked', 'fc1.weight', 'fc1.bias', 'fc_audioset.weight', 'fc_audioset.bias'])
key_needed = np.array(["bn0.weight", "bn0.bias", "bn0.running_mean", "bn0.running_var", "features.0.0.weight", "features.0.2.weight", "features.0.2.bias", "features.0.2.running_mean", "features.0.2.running_var", "features.1.0.weight", "features.1.2.weight", "features.1.2.bias", "features.1.2.running_mean", "features.1.2.running_var", "features.1.4.weight", "features.1.5.weight", "features.1.5.bias", "features.1.5.running_mean", "features.1.5.running_var", "features.2.0.weight", "features.2.2.weight", "features.2.2.bias", "features.2.2.running_mean", "features.2.2.running_var", "features.2.4.weight", "features.2.5.weight", "features.2.5.bias", "features.2.5.running_mean", "features.2.5.running_var", "features.3.0.weight", "features.3.2.weight", "features.3.2.bias", "features.3.2.running_mean", "features.3.2.running_var", "features.3.4.weight", "features.3.5.weight", "features.3.5.bias", "features.3.5.running_mean", "features.3.5.running_var", "features.4.0.weight", "features.4.2.weight", "features.4.2.bias", "features.4.2.running_mean", "features.4.2.running_var", "features.4.4.weight", "features.4.5.weight", "features.4.5.bias", "features.4.5.running_mean", "features.4.5.running_var", "features.5.0.weight", "features.5.2.weight", "features.5.2.bias", "features.5.2.running_mean", "features.5.2.running_var", "features.5.4.weight", "features.5.5.weight", "features.5.5.bias", "features.5.5.running_mean", "features.5.5.running_var", "features.6.0.weight", "features.6.2.weight", "features.6.2.bias", "features.6.2.running_mean", "features.6.2.running_var", "features.6.4.weight", "features.6.5.weight", "features.6.5.bias", "features.6.5.running_mean", "features.6.5.running_var", "features.7.0.weight", "features.7.2.weight", "features.7.2.bias", "features.7.2.running_mean", "features.7.2.running_var", "features.7.4.weight", "features.7.5.weight", "features.7.5.bias", "features.7.5.running_mean", "features.7.5.running_var", "features.8.0.weight", "features.8.2.weight", "features.8.2.bias", "features.8.2.running_mean", "features.8.2.running_var", "features.8.4.weight", "features.8.5.weight", "features.8.5.bias", "features.8.5.running_mean", "features.8.5.running_var", "features.9.0.weight", "features.9.2.weight", "features.9.2.bias", "features.9.2.running_mean", "features.9.2.running_var", "features.9.4.weight", "features.9.5.weight", "features.9.5.bias", "features.9.5.running_mean", "features.9.5.running_var", "features.10.0.weight", "features.10.2.weight", "features.10.2.bias", "features.10.2.running_mean", "features.10.2.running_var", "features.10.4.weight", "features.10.5.weight", "features.10.5.bias", "features.10.5.running_mean", "features.10.5.running_var", "features.11.0.weight", "features.11.2.weight", "features.11.2.bias", "features.11.2.running_mean", "features.11.2.running_var", "features.11.4.weight", "features.11.5.weight", "features.11.5.bias", "features.11.5.running_mean", "features.11.5.running_var", "features.12.0.weight", "features.12.2.weight", "features.12.2.bias", "features.12.2.running_mean", "features.12.2.running_var", "features.12.4.weight", "features.12.5.weight", "features.12.5.bias", "features.12.5.running_mean", "features.12.5.running_var", "features.13.0.weight", "features.13.2.weight", "features.13.2.bias", "features.13.2.running_mean", "features.13.2.running_var", "features.13.4.weight", "features.13.5.weight", "features.13.5.bias", "features.13.5.running_mean", "features.13.5.running_var", "fc1.weight", "fc1.bias", "fc_audioset.weight", "fc_audioset.bias"])

key_to_suppress = np.setdiff1d(key_before, key_needed)

key_not_to_suppress = np.array(["bn0.num_batches_tracked", "features.0.2.num_batches_tracked", "features.1.2.num_batches_tracked", "features.1.5.num_batches_tracked", "features.2.2.num_batches_tracked", "features.2.5.num_batches_tracked", "features.3.2.num_batches_tracked", "features.3.5.num_batches_tracked", "features.4.2.num_batches_tracked", "features.4.5.num_batches_tracked", "features.5.2.num_batches_tracked", "features.5.5.num_batches_tracked", "features.6.2.num_batches_tracked", "features.6.5.num_batches_tracked", "features.7.2.num_batches_tracked", "features.7.5.num_batches_tracked", "features.8.2.num_batches_tracked", "features.8.5.num_batches_tracked", "features.9.2.num_batches_tracked", "features.9.5.num_batches_tracked", "features.10.2.num_batches_tracked", "features.10.5.num_batches_tracked", "features.11.2.num_batches_tracked", "features.11.5.num_batches_tracked", "features.12.2.num_batches_tracked", "features.12.5.num_batches_tracked", "features.13.2.num_batches_tracked", "features.13.5.num_batches_tracked"])

key_to_suppress = np.setdiff1d(key_to_suppress,key_not_to_suppress)




for key in key_to_suppress:
    if key in checkpoint['model']:
        print("erase key :",checkpoint['model'][key])
        del checkpoint['model'][key]

# key_add_suppress = np.array(["bn0.num_batches_tracked", "features.0.2.num_batches_tracked", "features.1.2.num_batches_tracked", "features.1.5.num_batches_tracked", "features.2.2.num_batches_tracked", "features.2.5.num_batches_tracked", "features.3.2.num_batches_tracked", "features.3.5.num_batches_tracked", "features.4.2.num_batches_tracked", "features.4.5.num_batches_tracked", "features.5.2.num_batches_tracked", "features.5.5.num_batches_tracked", "features.6.2.num_batches_tracked", "features.6.5.num_batches_tracked", "features.7.2.num_batches_tracked", "features.7.5.num_batches_tracked", "features.8.2.num_batches_tracked", "features.8.5.num_batches_tracked", "features.9.2.num_batches_tracked", "features.9.5.num_batches_tracked", "features.10.2.num_batches_tracked", "features.10.5.num_batches_tracked", "features.11.2.num_batches_tracked", "features.11.5.num_batches_tracked", "features.12.2.num_batches_tracked", "features.12.5.num_batches_tracked", "features.13.2.num_batches_tracked", "features.13.5.num_batches_tracked"])




keys = np.array([])
keys = checkpoint["model"].keys()
print(keys)
for key in keys :
    print(key)



# model.load_state_dict(checkpoint['model'])


# # Test Model

# # res_height = 1 # need to put the good dimensions (need to be fix but i want it to adapt for the right size)
# # res_width = 224000
# # batch_size = 16
# # summary(model, input_size=(,res_height, res_width))

path_model = "../audioset_tagging_cnn/MobileNetTens/Weight_NN/LeeTens_10_empty.onnx"
Convert_ONNX(path_model)

onnx_model = onnx.load(path_model)
model_simp, check = simplify(onnx_model)
onnx.save(model_simp, path_model)
# onnx_model = onnx.load(path_model)
# model_simp, check = simplify(onnx_model)
# onnx.save(model_simp, path_model)

# print("conversion complete")
