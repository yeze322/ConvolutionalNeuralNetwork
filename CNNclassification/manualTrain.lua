require 'xlua'

MAX_ITERATION = 3
LEARNING_RATE = 0.001
NET_NAME = 'cnnet.t7'

local LEN = trainset:size()

function gradUpdate(x, y)
   local pred = net:forward(x)
   local loss = criterion:forward(pred, y)
   net:zeroGradParameters()
   t = criterion:backward(pred, y)
   net:backward(x, t)
   net:updateParameters(LEARNING_RATE)
   return loss
end

function StartTrain()
    print('start training ... ' .. 'Result will be saved at: ' .. NET_NAME)
    print('maxIteration = ' .. MAX_ITERATION .. "\tLearning Rate = " .. LEARNING_RATE)
    local losses = torch.Tensor(LEN)
    for iter=1, MAX_ITERATION do
        for i=1,LEN do
            xlua.progress(i,Len)
            losses[i] = gradUpdate(trainset.data[i], trainset.label[i])
        end
        print("Iter: " .. iter, "Loss: " .. losses:mean())
    end
    print('FInished trainning!')
    torch.save(NET_NAME, net)
    print('Net saved!' .. 'File name: ' .. NET_NAME)
end

print('Function loaded: StartTrain()')
