"""
Vertically partitioned SplitNN implementation

Worker 1 has two segments of the model and the Images

Worker 2 has a segment of the model and the Labels
"""


class SplitNN:
    def __init__(self, models, optimizers):
        self.models = models
        self.optimizers = optimizers

    def forward(self, x):
        grads = []
        remote_grads = []

        grads.append(models[0](x))
        if grads[-1].location == models[1].location:
            remote_grads.append(grads[-1].detach().requires_grad_())
        else:
            remote_grads.append(
                grads[-1].detach().move(models[1].location).requires_grad_()
            )
        i = 1
        while i < (len(models) - 1):

            grads.append(models[i](remote_grads[-1]))
            if grads[-1].location == models[i + 1].location:
                remote_grads.append(grads[-1].detach().requires_grad_())
            else:
                remote_grads.append(
                    grads[-1].detach().move(models[i + 1].location).requires_grad_()
                )
            i += 1
        grads.append(models[i](remote_grads[-1]))
        self.grads = grads
        self.remote_grads = remote_grads

        return grads[-1]

    def backward(self):
        grads = self.grads
        remote_grads = self.remote_grads
        optimizers = self.optimizers

        i = len(models) - 2
        while i > -1:
            if remote_grads[i].location == grads[i].location:
                grad_grads = remote_grads[i].grad.copy()
            else:
                grad_grads = remote_grads[i].grad.copy().move(grads[i].location)
            grads[i].backward(grad_grads)
            i -= 1

    def zero_grads(self):
        for opt in optimizers:
            opt.zero_grad()

    def step(self):
        for opt in optimizers:
            opt.step()


import torch
from torchvision import datasets, transforms
from torch import nn, optim
import syft as sy

hook = sy.TorchHook(torch)


torch.manual_seed(0)


# Define our model segments

input_size = 784
hidden_sizes = [128, 640]
output_size = 10

models = [
    nn.Sequential(nn.Linear(input_size, hidden_sizes[0]), nn.ReLU(),),
    nn.Sequential(nn.Linear(hidden_sizes[0], output_size), nn.LogSoftmax(dim=1)),
]

# # Send Model Segments to model locations
# model_locations = [worker1, worker2]
# for model, location in zip(models, model_locations):
#     model.send(location)
