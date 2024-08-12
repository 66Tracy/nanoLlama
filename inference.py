import os
import torch
from transformers import AutoTokenizer
from modeling_llama import LlamaCasualModel, LlamaConfig
from contextlib import nullcontext
ctx = nullcontext()

# inference config
# cuda:0 or cpu
device = 'cuda:0'
out_dir = 'out-shakespeare'

# init from a model saved in a specific directory
ckpt_path = os.path.join(out_dir, 'ckpt.pt')
checkpoint = torch.load(ckpt_path, map_location=device)
model_cfg = checkpoint['model_cfg']
model_cfg.device = device
model = LlamaCasualModel(model_cfg)
state_dict = checkpoint['model']
unwanted_prefix = '_orig_mod.'
for k,v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
model.load_state_dict(state_dict)

# begin inference
model.eval()
model.to(device)

num_samples = 10 # number of samples to draw
max_new_tokens = 500 # number of tokens generated in each sample
temperature = 0.8 # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
top_k = 200 # retain only the top_k most likely tokens, clamp others to have 0 probability

prompt = 'GRUMIO:'
tokenizer = AutoTokenizer.from_pretrained("./gpt2-tokenizer")
x = tokenizer(prompt, return_tensors="pt").input_ids
x = x.to(device)

# run generation
with torch.no_grad():
    with ctx:
        for k in range(num_samples):
            y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
            print(tokenizer.decode(y[0].tolist()))
            print('---------------')
