# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

import numpy as np

from HelperClass.NeuralNet_1_1 import *

from HelperClass.NeuralNet_1_1 import *
from HelperClass.DataReader_1_1 import *
from HelperClass.HyperParameters_1_0 import *

file_name = 'D:/housing.npz'

# main
if __name__ == '__main__':
    # data
    reader = DataReader_1_1(file_name)
    reader.ReadData()
    reader.NormalizeX()
    reader.NormalizeY()
    # net
    hp = HyperParameters_1_0(13, 1, eta=0.001, max_epoch=1000, batch_size=20, eps=1e-5)
    net = NeuralNet_1_1(hp)
    net.train(reader, checkpoint=0.1)
    # inference
    x1=0.09744
    x2=0.00
    x3=5.960    
    x4=0
    x5=0.4990
    x6=5.8410
    x7=61.40
    x8=3.3779
    x9=5
    x10=279.0
    x11=19.20
    x12=377.560
    x13=11.41
    #standard output=20.0
    x=np.array([x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13]).reshape(1,13)
    x_new=reader.NormalizePredicateData(x)
    z=net.inference(x_new)
    print("z=", z)
    z_true=z*reader.Y_norm[0,1] + reader.Y_norm[0,0]    
    print("Z_true=", Z_true)
