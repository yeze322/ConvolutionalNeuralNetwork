mean = {} -- store the mean, to normalize the test set in the future
stdv  = {} -- store the standard-deviation for the future

print('\nNormalizing...')
-- 1 - trainset normalization
for i=1,1 do
    mean[i] = trainset.data[{ {}, {i}, {}, {}  }]:mean() -- mean estimation
    print('Channel ' .. i .. ', Mean: ' .. mean[i])
    trainset.data[{ {}, {i}, {}, {}  }]:add(-mean[i]) -- mean subtraction
    
    stdv[i] = trainset.data[{ {}, {i}, {}, {}  }]:std() -- std estimation
    print('Channel ' .. i .. ', Standard Deviation: ' .. stdv[i])
    trainset.data[{ {}, {i}, {}, {}  }]:div(stdv[i]) -- std scaling
end


-- 2 - testset normalization
for i=1,1 do
    testset.data[{{},{i},{},{}}]:add(-mean[i])
    testset.data[{{},{i},{},{}}]:div(stdv[i])
end
print('testset normalization finished!\n')
