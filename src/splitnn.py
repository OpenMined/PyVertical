"""
Vertically partitioned SplitNN implementation

Alice worker has two segments of the model and the Images

Bob worker has a segment of the model and the Labels
"""


import torch
from torchvision import datasets, transforms
from torch import nn, optim
import syft as sy
hook = sy.TorchHook(torch)

class SplitNN:
    def __init__(self, models, optimizers):
        self.models = models
        self.optimizers = optimizers
        
    def forward(self, x):
        a = []
        remote_a = []
        
        a.append(models[0](x))
        if a[-1].location == models[1].location:
            remote_a.append(a[-1].detach().requires_grad_())
        else:
            remote_a.append(a[-1].detach().move(models[1].location).requires_grad_())

        i=1    
        while i < (len(models)-1):
            
            a.append(models[i](remote_a[-1]))
            if a[-1].location == models[i+1].location:
                remote_a.append(a[-1].detach().requires_grad_())
            else:
                remote_a.append(a[-1].detach().move(models[i+1].location).requires_grad_())
            
            i+=1
        
        a.append(models[i](remote_a[-1]))
        self.a = a
        self.remote_a = remote_a
        
        return a[-1]
    
    def backward(self):
        a=self.a
        remote_a=self.remote_a
        optimizers = self.optimizers
        
        i= len(models)-2   
        while i > -1:
            if remote_a[i].location == a[i].location:
                grad_a = remote_a[i].grad.copy()
            else:
                grad_a = remote_a[i].grad.copy().move(a[i].location)
            a[i].backward(grad_a)
            i-=1

    
    def zero_grads(self):
        for opt in optimizers:
            opt.zero_grad()
        
    def step(self):
        for opt in optimizers:
            opt.step()
    
    torch.manual_seed(0)

# Define our model segments

input_size = 784
hidden_sizes = [128, 640]
output_size = 10

models = [
    nn.Sequential(
                nn.Linear(input_size, hidden_sizes[0]),
                nn.ReLU(),
    ),
    nn.Sequential(
                nn.Linear(hidden_sizes[0], hidden_sizes[1]),
                nn.ReLU(),
    ),
    nn.Sequential(
                nn.Linear(hidden_sizes[1], output_size),
                nn.LogSoftmax(dim=1)
    )
]

# Create optimisers for each segment and link to them
optimizers = [
    optim.SGD(model.parameters(), lr=0.03,)
    for model in models
]

# create some workers
alice = sy.VirtualWorker(hook, id="alice")
bob = sy.VirtualWorker(hook, id="bob")

# Send Model Segments to model locations
model_locations = [alice, alice, bob]
for model, location in zip(models, model_locations):
    model.send(location)

#Instantiate a SpliNN class with our distributed segments and their respective optimizers
splitNN = SplitNN(models, optimizers)