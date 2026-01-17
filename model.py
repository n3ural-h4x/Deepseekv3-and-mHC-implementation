#IMPLEMENTATION OF THE DEEPSEEK mHC and v3:
import torch
import torch.nn.functional as F 
import torch.nn as nn

class RMSnorm(nn.Module):
  def __init__(self, eps=1e-6):
    super().__init__()
    self.eps = eps
  def forward(self, x:torch.Tensor):
    return x / torch.sqrt(torch.mean(torch.square(x), dim=-1, keepdim=True) + self.eps) #keep the last dim
  
class Rope(nn.Module):
    def __init__(self, seq_len:int, d_model:int, theta=10000):
     super().__init__()
     self.length = torch.arange(0, seq_len).float()[:, None]
     self.dim = torch.pow(theta, torch.arange(0, d_model, 2).float() / d_model)[None, :]     
     self.matrix = torch.mul(self.length, 1.0 / self.dim)
     self.register_buffer('matrix', self.length * 1.0 / self.dim, persistent=False)

    def calculate(self):
     self.cos = torch.cos(self.matrix)
     self.sin = torch.sin(self.matrix)
     return torch.stack([self.cos, self.sin], dim=-1)
    
    def forward(self, x:torch.Tensor, head_dimension:torch.Tensor, dim:int):
      seq_len = x.shape[2]
      x1, x2 = x[..., :dim//2], x[..., dim//2:]
      head_dimension = head_dimension[:seq_len, :]
      head_dimension = head_dimension[None, None, :, :]
      x_cos, x_sin = torch.split(head_dimension,  dim=-1)
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
    # for Q query vector
    c_Q = self.W_dq(x)
    q_c = self.W_uq(c_Q).view(b, seq_len, self.n_heads, self.d_head).transpose(1, 2)
    c_Q = self.W_qr(c_Q).view(b, seq_len, self.n_heads, self.d_head).transpose(1, 2)
    #TODO FOR THE Q_R YOU HAVE TO CHANGE THE SPECIFIC DIM
    #DONE
    Q_r = self.rope(head_dimension, c_Q)
    Q = torch.cat([Q_r, q_c], dim=-1)

    # Computation 
    attn_matrix  = Q @ K.transpose(-1, -2) / torch.sqrt()

    causal_mask  = torch.ones_like(attn_matrix) 

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
  def __init__(self, top_k:int, n_experts:int, d_model:int, n_groups:int, bias:bool, top_k_groups:int):
    super().__init__()
    self.top_k = top_k
    self.top_k_groups = top_k_groups
    self.n_experts = n_experts
    self.router = nn.Linear(d_model, n_experts)
    self.n_groups = n_groups
    self.bias = torch.empty(n_experts) if bias else None

  def forward(self, x:torch.Tensor):
    #batch, seq_len, d_model = x.size()
    x_routed = self.router(x)
    x_flat = x_routed(-1, self.n_experts)
    x_flat = x_flat.sigmoid()
    og_scores = x_flat
    x_flat += self.bias #Bias term for way around balance loss
    #top_k_weight, top_k_idx = torch.top_k(x_flat, dim=-1, k=self.top_k)
    if self.n_groups > 1:
      x_flat = x_flat.view(x_flat.size(0), self.n_groups, -1) # MAKE IT VIEW INTO GRPS
      if self.bias is None:
        grp_scores = x_flat.amax(dim=-1)
      else:
        grp_scores = x_flat.topk(2, dim=-1)[0].sum(dim=-1) #WHY DOES THEY USE TOP_K = 2 IDK WILL HAVE TO LOOK MORE INTO IT
      _, grp_idx = grp_scores.topk(self.topk_groups, dim=-1) # Now take the highest idx from the weight scores
      mask = x_flat.new_ones(x_flat.size(0), self.n_groups).scatter_(grp_idx, dim=-1)
      scores = x_flat.masked_fill_(mask[:, None], float(-'inf')).flatten(1)

    indices = torch.topk(scores, self.top_k, dim=-1)[1]
    weights = og_scores.gather(1, indices)
    weights *= self.route_scale
    return weights, indices

class MoE(nn.Module):
  def __init__(self,top_k, n_experts, d_model, n_group, bias, top_k_groups, d_ff, shared_experts):
    super().__init__()
    self.d_model = d_model
    self.router = Router(top_k, n_experts, d_model, n_group, bias, top_k_groups)
    self.experts = nn.Modulelist([SwigluFFN(d_model, d_ff) for _ in range(n_experts)])
    self.shared_experts = shared_experts
    self.d_ff = d_ff
    self.shared_experts = SwigluFFN(d_model, self.shared_experts*d_ff)
  def forward(self, x:torch.Tensor):
    weights, idx = self.router(x)
    y = torch.zeros_like(x.view(-1, self.d_model))
    total_count = torch.bincount(idx.view(-1), self.n_experts).to_list()
    for i in range(self.n_experts):
      if total_count[i] == 0:
        continue
      expert = self.experts[i]
      indicies, top = torch.where(idx == i) #gives the x,y coordinates
      y[indicies] += expert(x[indicies]) * weights[indicies, top, None] # take the indices which will have the following experts and add the weights to it
    z = self.shared_experts(x)
    return (z+y).view_as(x)
  
class TransformerBlock(nn.Module):
  def __init__(self,use_moe:bool):
    super().__init__()
    self.use_moe = use_moe
    self.rms1 = RMSnorm()
    self.rms2 = RMSnorm()
    self.mla = MLA()
    self.ffn = MoE() if self.use_moe else SwigluFFN()
  def forward(self, x:torch.Tensor):
    x = x + self.mla(self.rms1(x))
    x = x + self.ffn(self.rms2(x))
    return x
  
class DeepSeekv3(nn.Module):
  def __init__(self,layers:int ):
    super().__init__()
    self.layers = layers
    self.layers = nn.Modulelist([TransformerBlock() for _ in range(self.layers)])
    self.lm_head = nn.Linear(d_model, num_of_tokens)
    self.embedding = nn.embedding
  
  def forward(self, x:torch.Tensor):
    @torch.no_grad():
    @torch.no_grad
































 


     

     
