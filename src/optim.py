from abc import abstractmethod
from typing import List

from .graph import Node

class Optimizer:
    def __init__(self, params: List[Node], lr: float):
        self.params = params
        self.lr = lr

    def zero_grad(self):
        for param in self.params:
            param.grad = 0

    @abstractmethod
    def step(self):
        raise NotImplementedError

class SGD(Optimizer):
    def __init__(self, params: List[Node], lr=0.001, momentum=0, weight_decay=0.0, nesterov=False):
        super().__init__(params, lr)
        self.momentum, self.wd, self.nesterov = momentum, weight_decay, nesterov
        self.b = [0. for _ in range(len(self.params))] if self.momentum else []

    def step(self) -> None:
        for i, param in enumerate(self.params):
            g = param.grad + self.wd * param.value
            if self.momentum:
                self.b[i] = self.momentum * self.b[i] + g
                g = (g + self.momentum * self.b[i]) if self.nesterov else self.b[i]
            param.value -= g * self.lr
