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

    res_height = 1 # need to put the good dimensions (need to be fix but i want it to adapt for the right size)
    res_width = 32000
    
    x = torch.randn(1,1,res_height, res_width, device="cpu")
    print("input : ",x.shape)
    torch.onnx.export(
                model,# your pytorch model
                x, # need to change to put the right entrance
                path_model, # path to save
                export_params=True,
                opset_version=10, # mandatory for tensil 10 by default
                input_names=["Input"],
                output_names=["Output"],
    )

classes_num = 527
model = LeeNet11(classes_num)
path = "../audioset_tagging_cnn/MobileNetTens/LeeNet11.pth"
checkpoint = torch.load(path, map_location=torch.device('cpu'))


#Match corresponding dimension in 1D to 2D
for i in range(1,10) :
    checkpoint["model"]["conv_block"+str(i)+".conv1.weight"] =  torch.unsqueeze(checkpoint["model"]["conv_block"+str(i)+".conv1.weight"],dim=2)

model.load_state_dict(checkpoint['model'])


# # Test Model

# # res_height = 1 # need to put the good dimensions (need to be fix but i want it to adapt for the right size)
# # res_width = 224000
# # batch_size = 16
# # summary(model, input_size=(,res_height, res_width))


# Conversion to ONNX
path_model = "../audioset_tagging_cnn/MobileNetTens/Weight_NN/LeeTens.onnx"
Convert_ONNX(path_model)

#if we need to have a empty onnx

# onnx_model = onnx.load(path_model)
# model_simp, check = simplify(onnx_model)
# onnx.save(model_simp, path_model)

print("conversion complete")
