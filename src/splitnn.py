"""
Vertically partitioned SplitNN implementation

Worker 1 has two segments of the model and the Images

Worker 2 has a segment of the model and the Labels
"""


import syft as sy
import torch
from torch import nn

hook = sy.TorchHook(torch)


class SplitNN:
    def __init__(self, models, optimizers):
        self.models = models
        self.optimizers = optimizers

        self.data = []
        self.remote_tensors = []

    def forward(self, x):
        data = []
        remote_tensors = []

        data.append(models[0](x))

        if data[-1].location == models[1].location:
            remote_tensors.append(data[-1].detach().requires_grad_())
        else:
            remote_tensors.append(
                data[-1].detach().move(models[1].location).requires_grad_()
            )

        i = 1
        while i < (len(models) - 1):
            data.append(models[i](remote_tensors[-1]))

            if data[-1].location == models[i + 1].location:
                remote_tensors.append(data[-1].detach().requires_grad_())
            else:
                remote_tensors.append(
                    data[-1].detach().move(models[i + 1].location).requires_grad_()
                )

            i += 1

        data.append(models[i](remote_tensors[-1]))

        self.data = data
        self.remote_tensors = remote_tensors

        return data[-1]

    def backward(self):
        data = self.data
        remote_tensors = self.remote_tensors

        i = len(models) - 2
        while i > -1:
            if remote_tensors[i].location == data[i].location:
                grads = remote_tensors[i].grad.copy()
            else:
                grads = remote_tensors[i].grad.copy().move(data[i].location)

            data[i].backward(grads)
            i -= 1

    def zero_grads(self):
        for opt in self.optimizers:
            opt.zero_grad()

    def step(self):
        for opt in self.optimizers:
            opt.step()


# Define our model segments

INPUT_SIZE = 784
hidden_sizes = [128, 640]
OUTPUT_SIZE = 10

models = [
    nn.Sequential(
        nn.Linear(INPUT_SIZE, hidden_sizes[0]),
        nn.ReLU(),
        nn.Linear(hidden_sizes[0], hidden_sizes[1]),
        nn.ReLU(),
    ),
    nn.Sequential(nn.Linear(hidden_sizes[1], OUTPUT_SIZE), nn.LogSoftmax(dim=1)),
]

# # Send Model Segments to model locations
# model_locations = [worker1, worker2]
# for model, location in zip(models, model_locations):
#     model.send(location)
