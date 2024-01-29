import asyncio

import torch
from miner import forward
from neurons.protocol import Task404


asyncio.run(
    forward(Task404(prompt="A Golden Poison Dart Frog"), torch.device("cuda:0"), ".")
)
