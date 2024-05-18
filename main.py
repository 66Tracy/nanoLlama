import torch
from model import LlamaConfig, LlamaCasualModel

print("This is start...")

config = LlamaConfig()
model = LlamaCasualModel(config=config)

input = torch.ones(1,2).int()
print("input:", input)
output = model(input)

w = model.get_weight_params() / 1024 /1024
print(w," Mb")