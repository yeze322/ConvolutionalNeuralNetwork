-- testset.data = testset.data:double()   -- convert from Byte tensor to Double tensor

--[[
for i=1,1 do -- over each image channel
    testset.data[{ {}, {i}, {}, {}  }]:add(-mean[i]) -- mean subtraction    
    testset.data[{ {}, {i}, {}, {}  }]:div(stdv[i]) -- std scaling
end
]]--

function __Check(x, y)
    local prediction = net:forward(x)
    local confidences, indices = torch.sort(prediction, true)  -- true means sort in descending order
    if y == indices[1] then
        return true
    else
        return false
    end
end

function Check(dataset)
    local correct = 0
    local length = dataset:size()
    for i=1,length do
        if __Check(dataset.data[i], dataset.label[i]) then correct = correct + 1 end
        if i%5000 == 0 then print("Progress: " .. i) end
    end
    print("Check Finished! " .. correct .. "/" .. length)
    print("Accuracy: " .. 100*correct/length .. "%")
end

print('Function loaded: [prediction_index]Check(cnnet, testcase)')
