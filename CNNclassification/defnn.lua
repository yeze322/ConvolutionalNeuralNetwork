require 'nn'
require 'math'

-- data parameters
CH_NUM = 3; -- channel number
PIC_SIZE = 32; -- input square picture matrices' size
OUT_PREDICT_NUM = 10; -- output predict result number

-- NN parameters
FTMAP_NUM_1 = 6; -- number of feature maps in feature layer 1
CONV_SIZE_1 = 5; -- convolution core size
FTMAP_NUM_2 = 16;
CONV_SIZE_2 = 5;

--def nn
net = nn.Sequential();
net:add(nn.SpatialConvolution(CH_NUM, FTMAP_NUM_1, CONV_SIZE_1, CONV_SIZE_1)); -- Layer 1
PIC_SIZE = PIC_SIZE - CONV_SIZE_1 + 1;

net:add(nn.SpatialMaxPooling(2,2,2,2)) -- Layer 2
PIC_SIZE = math.floor(PIC_SIZE/2);

net:add(nn.SpatialConvolution(FTMAP_NUM_1, FTMAP_NUM_2, CONV_SIZE_2, CONV_SIZE_2)) -- Layer 3
PIC_SIZE = PIC_SIZE - CONV_SIZE_2 + 1;

net:add(nn.SpatialMaxPooling(2,2,2,2)) -- Layer 4
PIC_SIZE = math.floor(PIC_SIZE/2);

net:add(nn.View(FTMAP_NUM_2*PIC_SIZE*PIC_SIZE)) -- Layer 5 : reshapes from a 3D tensor of 16x5x5 into 1D tensor of 16*5*5
net:add(nn.Linear(FTMAP_NUM_2 * PIC_SIZE * PIC_SIZE, 120)) -- layer 6 : fully connected layer (matrix multiplication between input and weights)
net:add(nn.Linear(120, 84))
net:add(nn.Linear(84, OUT_PREDICT_NUM))
net:add(nn.LogSoftMax()) -- converts the output to a log-probability. Useful for classification problems

-- define criterion
criterion = nn.ClassNLLCriterion()
print('CNN define finished!\n')
