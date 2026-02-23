from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import torch.utils.checkpoint
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import time
import copy
import tiktoken
import os
import sys
if os.name == 'nt':
  sys.stdout.reconfigure(encoding='utf-8')

gpu_name = torch.cuda.get_device_name(0)
compute_capability = torch.cuda.get_device_capability(0)
'''
Model based off GPT-2
Implements: 
  -RoPE
  -Flash attention v2
  -tied weights between first and last layer
  -register buffer mask
  -lower precision for faster computes 'torch.set_float32_matmul_precision('high')'
  -'fused' optimizer
  -gradient clipping
  -torch autocast
  -triton compilation (not on windows) model = torch.compile(model)
  -SwiGLU gated activation
  -LayerNorm > RMSNorm according to my testing, remember to test it on A100 when training for real!
  -Gradient checkpointing
  -Exponential Moving Average of weights
  -Multi-query attention or grouped-query attention (MQA/GQA)
  -Head-scaling

  TO IMPLEMENT:
    -multiple GPUs in parallel                                    MEDIUM
    -Fused Kernels / black-sparge tricks (idk what this is)
    -Block-sparse attention                                       MEDIUM
    -Freezing Embeddings

  OTHER FEATURES:
    -Mixture of Experts (MoE)
    -Speculative decoding                                         LOW
    -Sparse transformer routing
    -KV cache during generation                                   LOW
    -Check HellaSwag Eval
    -Check Eleuther Eval harness

'''

gpt2_base = tiktoken.get_encoding("gpt2")
enc = tiktoken.Encoding(
    name="gpt2_custom",
    pat_str=gpt2_base._pat_str,
    mergeable_ranks=gpt2_base._mergeable_ranks,
    special_tokens={
        **gpt2_base._special_tokens,
        "<USER>": 50257,
        "<BOT>": 50258,
        "<SYSTEM>": 50259,
    }
)
allowed = {"<USER>", "<BOT>", "<SYSTEM>"}

with open('data/data.txt', 'r', encoding='utf-8') as f:
    text = f.read()
print("File read")

training_data = enc.encode(text, allowed_special=allowed)
training_data = torch.tensor(training_data, dtype=torch.long, device='cuda')
print("tokenized sequence")
print(f"{len(training_data)} tokens")

@dataclass
class Hyperparameters:
  batch_size: int = 8          # number of batches of input sequences
  context_size: int = 1024      # max sequence length
  vocab_size: int = 50260       # number of unique tokens
  num_transformers: int = 12    # number of transformer blocks in the model
  num_heads: int = 4           # Must be able to divide num_embd
  num_embd: int = 1600          # Must be divisible by num_heads
  num_kv_groups: int = num_heads# Number of groups heads are divided into for GQA
  dropout: float = 0.0          # Dropout
  learn_rate: int = 2e-4        # Step size when learning
  epochs: int = 3               # number of times we go through all training data
  device: str = "cuda"          # Devide the model runs on
  grad_cp: bool = True          # Determines whether we use gradient checkpointing
  log_every: int = 1            # Determines how often we print some info in the console
  save_every: int = 1000        # Determines how often we save the model
params = Hyperparameters()




class TokenDataset(Dataset):
  def __init__(self, data):
    self.data = data

  def __len__(self):
    return (len(self.data) - params.context_size - 1)

  def __getitem__(self, index):
    return self.data[index: index + params.context_size], self.data[index + 1: index + params.context_size + 1]

dataset = TokenDataset(training_data)
loader = DataLoader(dataset=dataset, batch_size=params.batch_size, shuffle=True)





def build_rope_cache(seq_len, dim, device):
    assert dim % 2 == 0, "Embedding dim must be even for RoPE"

    half_dim = dim // 2
    # Calculate inverse frequencies
    inv_freq = 1.0 / (10000 ** (torch.arange(0, half_dim).float() / half_dim))

    # Outer product: [seq_len, half_dim]
    positions = torch.arange(seq_len, dtype=torch.float)
    angles = torch.einsum('i,j->ij', positions, inv_freq)

    # Duplicate to get [seq_len, dim] (interleave for even/odd)
    sin = torch.sin(torch.cat([angles, angles], dim=-1)).to(device)
    cos = torch.cos(torch.cat([angles, angles], dim=-1)).to(device)

    return sin, cos





def rotate_half(x):
    # x shape: [..., dim]
    x1 = x[..., ::2]  # even indices
    x2 = x[..., 1::2]  # odd indices
    return torch.stack([-x2, x1], dim=-1).reshape_as(x)





def apply_rotary_pos_emb(q, k, sin, cos):
    # Assumes sin, cos shape: [seq_len, dim]
    # q, k shape: B,H,T,hD
    # Broadcast sin/cos to match q/k shape
    sin = sin[None, None, :, :]  # [1, 1, seq_len, dim]
    cos = cos[None, None, :, :]  # same shape

    # Apply rotation
    q_rot = (q * cos) + (rotate_half(q) * sin)
    k_rot = (k * cos) + (rotate_half(k) * sin)

    return q_rot, k_rot





class EMA:
  def __init__(self, model, device, decay=0.999):
    self.ema_model = copy.deepcopy(model)
    self.ema_model.eval()
    self.decay = decay
    self.device = device





class CausalSelfAttention(nn.Module):
  '''
  Causal Self Attention Block
  calculates Q,K,V for the input
  splits C across all heads
  calculates attention scores Q @ K
  scales values based off attention scores V @ attention
  unifies the heads again
  mixes the heads with the causal_proj output linear layer.
  returns the new output
  '''
  def __init__(self, config):
    super().__init__()
    assert config.num_embd % config.num_heads == 0 # Assure the number of embeddings can be split among heads
    self.config = config
    self.num_heads = config.num_heads
    self.num_embd = config.num_embd
    self.q_proj = nn.Linear(config.num_embd, config.num_embd)                                                   # Gets query values
    self.kv_proj = nn.Linear(config.num_embd, 2 * (config.num_embd // config.num_heads) * config.num_kv_groups) # Gets Key and Value values
    self.head_scale = nn.Parameter(torch.ones(self.num_heads))                                                  # Learnable head scaling values
    self.causal_proj = nn.Linear(config.num_embd, config.num_embd)                                              # Output projection
    self.register_buffer("sin", build_rope_cache(config.context_size, config.num_embd // config.num_heads, config.device)[0])
    self.register_buffer("cos", build_rope_cache(config.context_size, config.num_embd // config.num_heads, config.device)[1])

  def forward(self, input):
    B, T, C = input.size() # B,T,C -> batch, context_size, num_embd
    G = self.config.num_kv_groups
    H = self.num_heads
    d_head = C // H
    heads_per_kv = H // G

    q = self.q_proj(input).view(B, T, H, d_head).transpose(1, 2)  # (B, T, C) -> (B, T, H, hD) -> (B, H, T, hD)
    kv = self.kv_proj(input)                                      # (B, T, C) -> (B, T, 2 * hD * kv_groups) NOTE: 2 * hD * kv_groups == 2 * G * hD
    kv = kv.view(B, T, G, 2, d_head).permute(0, 2, 3, 1, 4)       # (B, T, 2 * G * hD) -> (B, T, G, 2, hD) -> (B, G, 2, T, hD)
    k, v = kv[:, :, 0], kv[:, :, 1]                               # (B, G, 2, T, hD) -> (B, G, T, hD) x2
    k = k.repeat_interleave(heads_per_kv, dim=1)                  # Repeat k for each group -> (B, H, T, hD)
    v = v.repeat_interleave(heads_per_kv, dim=1)                  # Repeat v for each group -> (B, H, T, hD)

    q, k = apply_rotary_pos_emb(q, k, self.sin[:T], self.cos[:T])
    out = F.scaled_dot_product_attention(q,k,v,is_causal=True, dropout_p=self.config.dropout if self.training else 0.0) # Flash Attention

    out = out * self.head_scale.view(1, H, 1, 1) # Scale output values my learned parameters for each head

    out = out.transpose(1, 2)           # B,H,T,hD -> B,T,H,hD (preparing to remerge all heads)
    out = out.contiguous().view(B,T,C)  # B,T,H,hD -> B,T,C (puts all hD side by side again (effectively H * hD))

    out = self.causal_proj(out)
    return out





class MLP(nn.Module):
  '''
  MLP/FFN Block
  SwiGLU implementation
  Applies 'Non Linearity' to the model
  this is done to each individual token
  '''
  def __init__(self, config):
    super().__init__()
    self.linear1 = nn.Linear(config.num_embd, 4 * config.num_embd)
    self.silu = nn.SiLU()
    self.linear2 = nn.Linear(2 * config.num_embd, config.num_embd)

  def forward(self, input):
    input = self.linear1(input)             # Expand the input (B,T,C*4)
    input1, input2 = input.chunk(2, dim=-1) # Divide the newly expanded embd dim in half (B,T,C4) -> (B,T,C2) & (B,T,C2)
    gate = self.silu(input2)                # Put 1 half through the SiLU gate
    input = gate * input1                   # Collapse the inputs back to num_embd size (B,T,C1) * (B,T,C2) -> (B,T,C) NOTE: this is element-wise multiplication, NOT Matrix multiplicationn
    input = self.linear2(input)             # project the gated activations (its just sort of better to do this, not super sure why)
    return input





class Transformer(nn.Module):
  '''
  Transformer Block
  Takes input, then normalizes it
  Calculates causal self attention on all tokens
  Adds residuals
  Normalizes the new input
  Puts each token through the MLP/FFN
  Adds residuals again to finish
  '''
  def __init__(self, config):
    super().__init__()
    self.layerNorm1 = nn.LayerNorm(config.num_embd)
    self.attention = CausalSelfAttention(config)
    self.layerNorm2 = nn.LayerNorm(config.num_embd)
    self.MLP = MLP(config)

  def forward(self, input):
    normal_input1 = self.layerNorm1(input)          # Normalize embeddings
    attention_input = self.attention(normal_input1) # Calculate Self-Attention
    residual_input1 = input + attention_input       # Add residuals to attention
    normal_input2 = self.layerNorm2(residual_input1)# Normalize embeddings again
    MLP_input = self.MLP(normal_input2)             # Puts the inputs through the MLP
    residual_input2 = residual_input1 + MLP_input   #Final output of the transformer
    return residual_input2





class Model(nn.Module):
  '''
  Defines the Model:
  Tokens turn to embeddings
  Positional data is calculated for the total number of tokens
  Decoder-only transformers calculate self attention
  embeddings are normalized
  Final linear layer calculates probabilities
  '''
  def __init__(self, config):
    super().__init__()
    self.config = config
    self.grad_cp = config.grad_cp

    self.transformer = nn.ModuleDict(dict(
      token_embd = nn.Embedding(config.vocab_size, config.num_embd),
      #pos_encoding = nn.Embedding(config.context_size, config.num_embd),
      transformers = nn.ModuleList([Transformer(config) for _ in range(config.num_transformers)]),
      layerNorm = nn.LayerNorm(config.num_embd),
    ))
    self.linear = nn.Linear(config.num_embd, config.vocab_size, bias=False)

    self.transformer.token_embd.weight = self.linear.weight # Shares weights between input and output layers
  
  def forward(self, input, targets=None):
    B, T = input.size()
    input = self.transformer.token_embd(input) # Get token embeddings B,T,C

    #Put the input through all transformers sequentially
    for block in self.transformer.transformers:
      if self.grad_cp and self.training:
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
          input = torch.utils.checkpoint.checkpoint(block, input, use_reentrant=True)
      else:
        input = block(input)

    input = self.transformer.layerNorm(input) # Normalize the embeds B,T,C
    logits = self.linear(input)               # Get logits B,T,vocab_size

    loss = None
    if targets is not None:
      loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
    return logits, loss




torch.set_float32_matmul_precision('high')
torch.backends.cuda.allow_fp16_bf16_reduction_math_sdp(True)
model = Model(params)
model.to(params.device)
#model = torch.compile(model)
#print(triton.__version__)
print(sum(p.numel() for p in model.parameters())/1e6, 'M parameters')
print(f"1 epoch = {len(loader)} batches")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA version: {torch.version.cuda}")
print(f"cuDNN available?: {torch.backends.cudnn.is_available()}")
print(f"Is CUDA available? {torch.cuda.is_available()}")
print(f"GPU: {gpu_name}")
print(f"Compute capability: {compute_capability}")
print(f"Built in FlashAttention?: {torch.backends.cuda.is_flash_attention_available()}")
print(f"FlashAttention enabled? {torch.backends.cuda.flash_sdp_enabled()}")
print(f"Memory effecient flashAttention enabled? {torch.backends.cuda.mem_efficient_sdp_enabled()}")
print(f"fp16/bf16 enabled?: {torch.backends.cuda.fp16_bf16_reduction_math_sdp_allowed()}")
#sys.exit(0)
optimizer = torch.optim.AdamW(model.parameters(), lr=params.learn_rate, fused=True)

#Training Loop
for epoch in range(params.epochs):
  for index, (inputs, targets) in enumerate(loader):
    t0 = time.time() 
    optimizer.zero_grad()
    inputs = inputs.to(params.device)
    targets = targets.to(params.device)
    with torch.autocast(device_type=params.device, dtype=torch.bfloat16):
      logits, loss = model(inputs, targets)
    loss.backward()
    norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    torch.cuda.synchronize()
    t1 = time.time()

    if (index % params.log_every == 0):
      dt = (t1 - t0)*1000
      tok_per_sec = (params.batch_size * params.context_size) / (t1 - t0)
      print(f"epoch {epoch}, iter {index}, norm {norm:.4f}, loss: {loss.item():.4f}, time: {dt:.2f}ms, tok/sec: {tok_per_sec:.2f}")
    
    if((index > 0 and index % params.save_every == 0)) or (index == len(loader)-1):
      torch.save({
        'epoch': epoch,
        'batch': index,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
      }, f'models/model8b_i{index}_l{loss.item():.4f}.pth')