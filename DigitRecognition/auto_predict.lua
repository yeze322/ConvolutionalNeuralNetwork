require 'csvigo';

function PredictAndSave(cnnet, dataset)
    local LEN = dataset:size()
    local result = {ImageId, Label}
    result.ImageId = {}
    result.Label = {}
    for i=1,LEN do
        local prediction = cnnet:forward(dataset.data[i])
        local confidences, indices = torch.sort(prediction, true)
        result.ImageId[i] = i;
        result.Label[i] = indices[1]
        if result.Label[i] == 10 then result.Label[i] = 0 end

        if i%2000 == 0 then print('progress: ' .. i .. '/' .. LEN) end
    end
    csvigo.save('prediction_testdata.csv', result)
    print('[Finished!] result saved in "./prediction_testdata.csv"')
    return result
end


print('Fucntion laoded: [result]PredictionAndSave(cnneet, dataset) from "auto_prediction.lua"!\n')
