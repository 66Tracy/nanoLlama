from typing import Callable, List, Optional, Tuple, Union
import math
import inspect
import torch
import torch.nn.functional as F
from torch.nn.modules.module import Module
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

## 1) Llama配置文件
class LlamaConfig():
    def __init__(
        self,
        vocab_size=50257,
        hidden_size=1024,
        intermediate_size=2048,
        num_hidden_layers=12,
        num_attention_heads=16,
        max_position_embeddings=512,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        rope_scaling=None,
        attention_bias=False,
        attention_dropout=0.0,
        mlp_bias=False,
        block_size=512,
        device='cuda'
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.mlp_bias = mlp_bias
        self.block_size = block_size
        self.device = device


## 2) LlamaRMSNorm
class RMSNorm(nn.Module):
    def __init__(self, hidden_size=1024, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))
    
    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(dim=-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)

        return self.weight * hidden_states.to(input_dtype)

## 3）RoPE
class LlamaRotaryPositionEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=512, base=10000, device=None, scaling_factor=1.0):
        super().__init__()
        self.scaling_factor = scaling_factor
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        ## [dim]
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, x, position_ids):
        # x: [bs, num_attention_heads, seq_len, head_size]
        # inv_freq: [head_size] -> [1, head_size, 1].to(torch.float32) -> [bs, head_size, 1]
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        # position_ids: [bs, seq_len] -> [bs, 1, seq_len].to(torch.float32)
        position_ids_expanded = position_ids[:, None, :].float()

        # Force float32 since bfloat16 loses precision on long contexts
        # See https://github.com/huggingface/transformers/pull/29285
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        # 这里的意思应该是不允许混合精度训练，因为enabled=False
        with torch.autocast(device_type=device_type, enabled=False):
            # [bs, head_size, seq_len] = [bs, head_size, 1] @ [bs, 1, seq_len] -> [bs, seq_len, head_size]
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            # emb: [bs, seq_len, head_size * 2]
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
        # [bs, seq_len, head_dim], [bs, seq_len, head_dim]
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


## 4）Attention
class LlamaAttention(nn.Module):

    def __init__(self, config: LlamaConfig, layer_idx:Optional[int]):
        super().__init__()

        ## 检验一下
        assert config.hidden_size % config.num_attention_heads == 0

        ## 变量准备
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_attention_heads
        self.attention_dropout = config.attention_dropout

        ## 映射准备
        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=self.config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=self.config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=self.config.attention_bias)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=self.config.attention_bias)

        ## 初始化rope
        self._init_rope()
    
    def _init_rope(self):
        self.rope = LlamaRotaryPositionEmbedding(self.head_dim, max_position_embeddings=self.config.max_position_embeddings, device=self.config.device, base=self.config.rope_theta)
    
    def forward(self, 
                hidden_states: torch.Tensor,
                position_ids: Optional[torch.LongTensor] = None,
                attention_mask: Optional[torch.Tensor] = None):
        
        bsz, seq_len, hidden_dim = hidden_states.shape

        ## [bsz,seq_len,hidden_dim] -> [bsz,num_attention_heads,seq_len,head_dim]
        query_states = self.q_proj(hidden_states).view(bsz, seq_len, self.num_attention_heads, self.head_dim).transpose(1,2)
        key_states = self.k_proj(hidden_states).view(bsz, seq_len, self.num_attention_heads, self.head_dim).transpose(1,2)
        value_states = self.v_proj(hidden_states).view(bsz, seq_len, self.num_attention_heads, self.head_dim).transpose(1,2)

        ## 位置编码信息
        cos, sin = self.rope(value_states, position_ids=position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        ## self-attention
        ## [bsz, num_attention_heads, seq_len, seq_len]
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
        if attention_mask is not None:  # no matter the length, we just slice it
            ## 原文只对最后一个维度修改，我改成修改最后两个维度
            causal_mask = attention_mask[:, :, : query_states.shape[-2], : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask
        ## 类型转换成float32在转回去
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = F.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        ## [bsz, num_attention_heads, seq_len, seq_len] -> [bsz, num_attention_heads, seq_len, head_dim]
        attn_outputs = torch.matmul(attn_weights, value_states)

        ## [bsz, num_attention_heads, seq_len, head_dim] -> [bsz, seq_len, hidden_dim]
        outputs = attn_outputs.transpose(1,2).contiguous().reshape(bsz, seq_len, hidden_dim)
        outputs = self.o_proj(outputs)

        return outputs


## 5）LlamaMLP
class LlamaMLP(nn.Module):

    def __init__(self, config:LlamaConfig, layer_idx: Optional[int]):
        super().__init__()

        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.mlp_bias = config.mlp_bias
        
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=self.mlp_bias)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=self.mlp_bias)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=self.mlp_bias)
        self.act_fn = nn.SiLU()

    def forward(self, hidden_states:torch.Tensor):
        outputs = self.down_proj(self.act_fn(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))
        return outputs


## 6) decoder
class LlamaDecoderLayer(nn.Module):

    def __init__(self, 
                config:LlamaConfig,
                layer_idx: Optional[int]):
        super().__init__()

        self.layer_idx = layer_idx
        self.config = config

        self.attn = LlamaAttention(config, layer_idx)
        self.mlp = LlamaMLP(config, layer_idx)
        self.input_layernorm = RMSNorm(hidden_size=config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(hidden_size=config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, 
                hidden_states:torch.Tensor,
                position_ids:Optional[torch.LongTensor],
                attention_mask:Optional[torch.Tensor]):
        
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.attn(hidden_states=hidden_states, position_ids=position_ids, attention_mask=attention_mask)
        hidden_states = hidden_states + residual

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = hidden_states + residual
        ## 以元组的形式返回
        outputs = (hidden_states,)

        return outputs

## 7) 羊驼模型
class LlamaCasualModel(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size

        ## 模型层次
        self.emb_tokens = nn.Embedding(self.vocab_size, self.hidden_size)
        self.layers = nn.ModuleList(
            [LlamaDecoderLayer(config=config, layer_idx=idx) for idx in range(config.num_hidden_layers)]
        )
        self.lm_head = nn.Linear(self.hidden_size, self.vocab_size, bias=False)
        self.norm = RMSNorm(hidden_size=self.hidden_size, eps=self.config.rms_norm_eps)

        ## 权重初始化
        self.apply(self._init_weights)

    ## 初始化权重
    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
    
    ## 返回模型参数
    def get_num_params(self):
        n_params = sum(p.numel() for p in self.parameters())
        return n_params
    
    ## 返回模型所占权重
    def get_weight_params(self):
        n_params = sum(p.numel()*p.element_size() for p in self.parameters())
        return n_params

    ## merged from transformers.models.llama.modeling_llama.LlamaModel._update_causal_mask
    def forward(self,
                input_ids: torch.Tensor,
                labels:Optional[torch.Tensor]=None,
                attention_mask:Optional[torch.Tensor]=None):
        
        ## 取出对应的词嵌入，与相应的定义
        input_tensor = self.emb_tokens(input_ids)
        dtype, device = input_tensor.dtype, input_tensor.device
        min_dtype = torch.finfo(dtype).min
        sequence_length = input_tensor.shape[1]
        target_length = (
            attention_mask.shape[-1]
            if attention_mask is not None
            else sequence_length + 1
        )

        ## 构建位置向量：[seq_Len] -> [1, seq_len] 
        position_ids = torch.arange(
            sequence_length, device=device
        ).unsqueeze(0)

        ## causal_mask: [seq_len, target_len]
        causal_mask = torch.full(
            (sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device
        )
        ## causal_mask: 转化为对角线为1的下三角矩阵: [seq_len, target_len]
        if sequence_length != 1:
            causal_mask = torch.triu(causal_mask, diagonal=1)
        causal_mask = causal_mask[None, None, :, :].expand(input_tensor.shape[0], 1, -1, -1)

        ## 此处用于处理外部传入的句子的padding掉末尾无效token的attention mask: [batch_size, seq_len]
        if attention_mask is not None and attention_mask.dim() == 2:
            causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
            mask_length = attention_mask.shape[-1]
            padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :]
            padding_mask = padding_mask == 0
            causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                padding_mask, min_dtype
            )

        # embed positions
        hidden_states = input_tensor
        ## 1）Docder传播
        for decoder_layer in self.layers:
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids
            )
        ## 1.1）输出是个元组，拿到隐变量输出
        hidden_states = layer_outputs[0]
        ## 2）norm
        hidden_states = self.norm(hidden_states)
        
        loss = None
        if labels is not None:
            ## 3）training: batch lm_head prediction
            logits = self.lm_head(hidden_states)
            logits = logits.float()
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-1)
            # # Standard Code: Shift so that tokens < n predict n
            # shift_logits = logits[..., :-1, :].contiguous()
            # shift_labels = labels[..., 1:].contiguous()
            # # Flatten the tokens
            # loss_fct = CrossEntropyLoss()
            # shift_logits = shift_logits.view(-1, self.config.vocab_size)
            # shift_labels = shift_labels.view(-1)
            # # Enable model parallelism
            # shift_labels = shift_labels.to(shift_logits.device)
            # loss = loss_fct(shift_logits, shift_labels)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(hidden_states[:, [-1], :]) # note: using list [-1] to preserve the time dim
            logits = logits.float()
            loss = None
            
        return logits, loss

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")

        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'

        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)

            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
