#IMPLEMENTATION OF THE DEEPSEEK mHC:
import torch
import torch.nn.functional as F 
import torch.nn as nn


class Rope(nn.Module):
    def __init__(self, seq_len:int, d_model:int, theta=10000):
     super().__init__()
     self.length = torch.arange(0, seq_len).float()[:, None]
     self.dim = torch.pow(theta, -torch.arange(0, d_model, 2).float() / d_model)[None, :]     
     self.matrix = torch.mul(self.length, self.dim)
     self.register_buffer('matrix', self.matrix)

    def calculate(self):
     self.cos = torch.cos(self.matrix)
     self.sin = torch.sin(self.matrix)
     return torch.stack([self.cos, self.sin], dim=-1)
    
    def forward(self, x:torch.Tensor, head_dimension:torch.Tensor):
      x1, x2 = x[..., 0::2], x[..., 1::2]
      head_dimension = head_dimension[None, None, :, :]
      x_cos, x_sin = torch.split(head_dimension, 2,  dim=-1)
      x_even = x1 * x_cos - x2 * x_sin
      x_odd= x1 * x_sin + x2 * x_cos
      return torch.cat([x_even, x_odd], dim=-1)
    

class MLA(nn.Module):
  def __init__(self, dim_in:int, seq_len:int, d_model:int, n_heads:int, latent_size:int):
    super().__init__()
    self.W_dkv = nn.Linear(dim_in, latent_size)
    self.W_uk = nn.Linear(latent_size, dim_in )
    self.W_kr = nn.Linear(d_model, latent_size)
    self.W_uv = nn.Linear(latent_size, d_model)
    self.n_heads = n_heads
    self.W_out = nn.Linear(dim_in, d_model)
    self.d_head = d_model // n_heads
    self.rope = Rope(seq_len, d_model)
    self.W_dq = nn.Linear(dim_in, latent_size)
    self.W_uq = nn.Linear(latent_size, d_model)
    self.W_qr = nn.Linear(d_model, latent_size)
  def forward(self, x:torch.Tensor):
    b, seq_len, d_model = x.size()
    C_dkv = self.W_dkv(x)
    K_C   = self.W_uk(C_dkv).view(b, seq_len, self.n_heads, self.d_head).transpose(1, 2)
    head_dimension = self.rope.calculate()
    k_r = self.W_kr(x).view(b, seq_len, 1, self.d_head).transpose(1, 2)
    k_r = self.rope(head_dimension, x=k_r)
    #todo us repeat interleave since kr has only 1 head dimension
    K = torch.concat([K_C, k_r], dim=-1)
    V = self.W_uv(C_dkv).view(b, seq_len, self.n_heads, self.d_head).transpose(1, 2)

    
    
    c_Q = self.W_dq(x)
    q_c = self.W_uq(c_Q).view(b, seq_len, self.n_heads, self.d_head).transpose(1, 2)
    c_Q = self.W_qr(c_Q).view(b, seq_len, self.n_heads, self.d_head).transpose(1, 2)
    #TODO FOR THE Q_R YOU HAVE TO CHANGE THE SPECIFIC DIM
    Q_r = self.rope(head_dimension, c_Q)
    Q = torch.concat([Q_r, q_c], dim=-1)

    #attn_matrix  = Q.tranpose(-1, -2) @ K / 


          
 


     

     
