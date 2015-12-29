require 'cunn'
net = net:cuda()
criterion = criterion:cuda()
trainset.label = trainset.label:cuda()
trainset.data = trainset.data:cuda()

testset.data = testset.data:cuda()
