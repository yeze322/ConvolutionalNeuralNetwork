trainer = nn.StochasticGradient(net, criterion)
trainer.learningRate = 0.05
trainer.maxIteration = 3 -- just do 5 epochs of training.
print('trainer define finshed!\n')
