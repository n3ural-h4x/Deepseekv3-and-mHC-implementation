#IMPLEMENTATION OF THE DEEPSEEK mHC:
import torch
import torch.nn.functional as F 
import torch.nn as nn


class Rope(nn.Module):
    def __init__(self, seq_len:int, d_model:int, theta=10000):
     super().__init__()
     self.length = torch.arange(0, seq_len).float()[:, None]
     self.dim = torch.pow(theta, torch.arange(0, d_model, 2).float() / d_model)[None, :]     
     self.matrix = torch.mul(self.length, 1.0 / self.dim)
     self.register_buffer('matrix', self.matrix)

    def calculate(self):
     self.cos = torch.cos(self.matrix)
     self.sin = torch.sin(self.matrix)
     return torch.stack([self.cos, self.sin], dim=-1)
    
    def forward(self, x:torch.Tensor, head_dimension:torch.Tensor, dim:int):
      seq_len = x.shape[2]
      x1, x2 = x[..., :dim//2], x[..., dim//2:]
      head_dimension = head_dimension[:seq_len, :]
      head_dimension = head_dimension[None, None, :, :]
      x_cos, x_sin = torch.split(head_dimension, 2,  dim=-1)
      x_even = x1 * x_cos - x2 * x_sin
      x_odd= x1 * x_sin + x2 * x_cos
      return torch.cat([x_even, x_odd], dim=-1)
    

class MLA(nn.Module):
  def __init__(self, dim_in:int, seq_len:int, d_model:int, n_heads:int, latent_size_q:int, latent_size_kv:int):
    super().__init__()
    #LATENT VECTORS SPECIFICALLY FOR THE KV
    self.W_dkv = nn.Linear(dim_in, latent_size_kv)
    self.W_uk = nn.Linear(latent_size_kv, d_model )
    self.W_kr = nn.Linear(d_model, latent_size_kv)
    self.W_uv = nn.Linear(latent_size_kv, d_model)
    self.n_heads = n_heads
    # D_HEAD IS THE SAME SINCE WE CONVERTING THE MATRIX BACK TO THE D_MODEL
    self.d_head = d_model // n_heads
    #USE THE ROPE CLASS HERE 
    self.rope = Rope(seq_len, d_model)
    #LATENT VECTOR SPECIFICALLY FOR Q
    self.W_dq = nn.Linear(dim_in, latent_size_q)
    self.W_uq = nn.Linear(latent_size_q, d_model)
    self.W_qr = nn.Linear(d_model, latent_size_q)
    self.W_out = nn.Linear(dim_in, d_model)
  def forward(self, x:torch.Tensor):
    b, seq_len, d_model = x.size()
    C_dkv = self.W_dkv(x) #(B, N, L_Q)
    K_C  = self.W_uk(C_dkv).view(b, seq_len, self.n_heads, self.d_head).transpose(1, 2)#(B, H, N, d_h)
    head_dimension = self.rope.calculate()
    k_r = self.W_kr(x).view(b, seq_len, 1, self.d_head).transpose(1, 2)
    k_r = k_r.expand(-1, self.n_heads, -1, -1)
    k_r = self.rope(head_dimension, x=k_r)
    #todo use repeat interleave since kr has only 1 head dimension
    K = torch.cat([K_C, k_r], dim=-1)
    V = self.W_uv(C_dkv).view(b, seq_len, self.n_heads, self.d_head).transpose(1, 2)

    
    
    c_Q = self.W_dq(x)
    q_c = self.W_uq(c_Q).view(b, seq_len, self.n_heads, self.d_head).transpose(1, 2)
    c_Q = self.W_qr(c_Q).view(b, seq_len, self.n_heads, self.d_head).transpose(1, 2)
    #TODO FOR THE Q_R YOU HAVE TO CHANGE THE SPECIFIC DIM
    Q_r = self.rope(head_dimension, c_Q)
    Q = torch.cat([Q_r, q_c], dim=-1)

    #attn_matrix  = Q.tranpose(-1, -2) @ K / 

class SwigluFFN(nn.Module):
  def __init__(self, d_model:int, d_ff:int):
    super().__init__()
    self.up_gate = nn.Linear(d_model, d_ff)
    self.gate = nn.Linear(d_model, d_ff)
    self.down_gate = nn.Linear(d_ff, d_model)
    self.activation = nn.SiLU()

  def forward(self, x:torch.Tensor):
    return self.down_gate(self.activation(self.up_gate(x)) * self.gate(x))

class Router(nn.Module):
  def __init__(self, top_k:int, n_experts:int, d_model:int):
    super().__init__()
    self.top_k = top_k
    self.n_experts = n_experts
    self.router = nn.Linear(d_model, n_experts)

  def forward(self, x:torch.Tensor):
    batch, seq_len, d_model = x.size()
    x_routed = self.router(x)
    x_flat = x_routed(-1, self.n_experts)
    x_flat = x_flat.sigmoid()
    top_k_weight, top_k_idx = torch.top_k(x_flat, dim=-1, k=self.top_k)
    router_weight = torch.empty



































 


     

     
