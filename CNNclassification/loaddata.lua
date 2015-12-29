trainset = torch.load('train.t7')
testset = torch.load('test.t7')
setmetatable(trainset, 
    {__index = function(t, i) 
                    return {t.data[i], t.label[i]} 
                end}
);

trainset.data = trainset.data:double() -- convert the data from a ByteTensor to a DoubleTensor.
testset.data = testset.data:double()

function trainset:size() 
    return self.data:size(1) 
end

function testset:size()
    return self.data:size(1)
end

print('Train Set: ', trainset)
print('Test Set: ', testset)
