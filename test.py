import numpy as np
import torch
from collections import OrderedDict

unsuper_trainData = np.load("Data-simple-45-22/pretraining_trainData.npy")
unsuper_trainLabel = np.load("Data-simple-45-22/StatLabel.npy")

print(type(unsuper_trainData))
print(unsuper_trainData.shape)
print(unsuper_trainData)

print("`````````````````````")

print(type(unsuper_trainLabel))
print(unsuper_trainLabel.shape)
print(unsuper_trainLabel)

 
pthfile = "simple-45.pth"            #.pth文件的路径
model = torch.load(pthfile, torch.device('cpu'))    #设置在cpu环境下查询
print(model)