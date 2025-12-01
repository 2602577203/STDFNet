import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_

from lib.models.layers.rpe import generate_2d_concatenated_self_attention_relative_positional_encoding_index

import math

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.,
                 rpe=False, z_size=7, x_size=14):
                 
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.rpe =rpe
        if self.rpe:
            relative_position_index = \
                generate_2d_concatenated_self_attention_relative_positional_encoding_index([z_size, z_size],
                                                                                           [x_size, x_size])
            self.register_buffer("relative_position_index", relative_position_index)
            # define a parameter table of relative position bias
            self.relative_position_bias_table = nn.Parameter(torch.empty((num_heads,
                                                                          relative_position_index.max() + 1)))
            trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self, x, mask=None, return_attention=False):
        # x: B, N, C
        # mask: [B, N, ] torch.bool
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale

        if self.rpe:
            relative_position_bias = self.relative_position_bias_table[:, self.relative_position_index].unsqueeze(0)
            attn += relative_position_bias

        if mask is not None:
            attn = attn.masked_fill(mask.unsqueeze(1).unsqueeze(2), float('-inf'),)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        if return_attention:
            return x, attn
        else:
            return x


class Attention_talking_head(nn.Module):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    # with slight modifications to add Talking Heads Attention (https://arxiv.org/pdf/2003.02436v1.pdf)
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.,
                 rpe=True, z_size=7, x_size=14):
        super().__init__()

        self.num_heads = num_heads

        head_dim = dim // num_heads

        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)

        self.proj = nn.Linear(dim, dim)

        self.proj_l = nn.Linear(num_heads, num_heads)
        self.proj_w = nn.Linear(num_heads, num_heads)

        self.proj_drop = nn.Dropout(proj_drop)

        self.rpe = rpe
        if self.rpe:
            relative_position_index = \
                generate_2d_concatenated_self_attention_relative_positional_encoding_index([z_size, z_size],
                                                                                           [x_size, x_size])
            self.register_buffer("relative_position_index", relative_position_index)
            # define a parameter table of relative position bias
            self.relative_position_bias_table = nn.Parameter(torch.empty((num_heads,
                                                                          relative_position_index.max() + 1)))
            trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self, x, mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1))

        if self.rpe:
            relative_position_bias = self.relative_position_bias_table[:, self.relative_position_index].unsqueeze(0)
            attn += relative_position_bias

        if mask is not None:
            attn = attn.masked_fill(mask.unsqueeze(1).unsqueeze(2),
                                    float('-inf'),)

        attn = self.proj_l(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        attn = attn.softmax(dim=-1)

        attn = self.proj_w(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x



# class MaskedCrossAttention(nn.Module):
#     #by ding : 这里proj_drop应该设置个值 0.1 不然不就为0了吗
#     def __init__(self, dim, hidden_dim, proj_drop=0.1):
#         super(MaskedCrossAttention, self).__init__()
 
#         self.linear_q = nn.Linear(dim, hidden_dim)
#         self.linear_k = nn.Linear(dim, hidden_dim)
#         self.linear_v = nn.Linear(dim, hidden_dim)
#         # self.linear_c = nn.Linear(dim, hidden_dim)
#         self.scale = dim ** -0.5
#         self.proj = nn.Linear(dim, dim)
#         self.proj_drop = nn.Dropout(proj_drop)
#         self.softplus = nn.Softplus()
 
#     def forward(self, x, mask, len_t):
        
#         template = x[:, :len_t]
#         search = x[:, len_t:]
#         template_TargetRelated = template * mask.unsqueeze(-1).expand(-1, -1, template.shape[2])
#         swapped_mask = torch.where(mask == 0, 1, 0)
#         template_TargetIrrelated = template * swapped_mask.unsqueeze(-1).expand(-1, -1, template.shape[2])
#         # print(mask[0].reshape(8,8))
#         s_v = self.linear_v(search)
#         mapped_s = self.linear_k(search)  
#         mapped_t1 = self.linear_q(template_TargetRelated)
#         mapped_t2 = self.linear_q(template_TargetIrrelated)
        
#         # mapped_t1 = mapped_t * mask.unsqueeze(-1).expand(-1, -1, template.shape[2])
#         # # mapped_t2 = self.linear_c(template) * swapped_mask.unsqueeze(-1).expand(-1, -1, template.shape[2])
#         # mapped_t2 = mapped_t * swapped_mask.unsqueeze(-1).expand(-1, -1, template.shape[2])
        
#         # 计算注意力权重  torch.Size([32, 256,64])
#         scores1 = torch.matmul(mapped_s, mapped_t1.transpose(1, 2))*self.scale  # (b,256,64)
#         scores2 = torch.matmul(mapped_s, mapped_t2.transpose(1, 2))*self.scale  # (b,256,64)
#         # print('score1',torch.mean(scores1, dim = 2)[0].max(dim=0))
#         # print('score2',torch.mean(scores2, dim = 2)[0].max(dim=0))
#         attn_TargetRelated = torch.exp(self.softplus(torch.sum(scores1, dim = 2)))# (b,256)
#         # v_max,_ = attn_TargetRelated.max(dim=1)
#         # v_min,_ = attn_TargetRelated.min(dim=1)
#         # weights = 0.0001 + (attn_TargetRelated - v_min.unsqueeze(-1)) / (v_max.unsqueeze(-1) - v_min.unsqueeze(-1)+0.0001)
#         attn_TargetIrrelated = torch.exp(self.softplus(torch.sum(scores2, dim = 2)))
#         # b_attn_TargetIrrelated_max,_ = attn_TargetIrrelated.max(dim=1)
#         # b_attn_TargetIrrelated_min,_ = attn_TargetIrrelated.min(dim=1)
#         # weights_b = 0.0001 + (attn_TargetIrrelated - b_attn_TargetIrrelated_min.unsqueeze(-1)) / (b_attn_TargetIrrelated_max.unsqueeze(-1) - b_attn_TargetIrrelated_min.unsqueeze(-1)+0.0001)

#         search = s_v * self.softplus(attn_TargetRelated.unsqueeze(-1) - attn_TargetIrrelated.unsqueeze(-1))
#         x = torch.cat([mapped_t1, search], dim=1)

        

#         x = self.proj(x)
#         x = self.proj_drop(x)
        
#         return x



class MaskedSpatialCrossAttention(nn.Module):

    def __init__(self, dim, proj_drop=0.):
        super(MaskedSpatialCrossAttention, self).__init__()
        self.proj_w = nn.Linear(256, 256)
        self.proj = nn.Linear(dim, dim)
        self.dim = dim
        self.scale = dim ** -0.5
        self.proj_drop = nn.Dropout(proj_drop)
        self.softplus = nn.Softplus()
    
    def Scale(self, x, scale=1):
        x_min,_ = x.min(dim=1)
        x_max,_ = x.max(dim=1)
        x_scaled = (x - x_min.unsqueeze(-1)) / (x_max.unsqueeze(-1) - x_min.unsqueeze(-1))
        x_scaled = x_scaled * scale
        return x_scaled
 
    def forward(self, x, xi, in_dict, mask, len_t):
        
        if in_dict["dynamic_template"] is not None:
            mask = torch.cat([mask, mask, in_dict["dynamic_template_mask"]], dim=1)
            template = x[:, :len_t*3]
            search = x[:, len_t*3:len_t*3+256]
            templatei = xi[:, :len_t*3]
            searchi = xi[:, len_t*3:len_t*3+256]
        else:
            mask = torch.cat([mask, mask], dim=1)
            template = x[:, :len_t*2]
            search = x[:, len_t*2:len_t*2+256]
            templatei = xi[:, :len_t*2]
            searchi = xi[:, len_t*2:len_t*2+256]
        
        q = template
        k = search
        v = search
        qi = templatei
        ki = searchi
        vi = searchi
        
        
        
        TargetRelated = torch.empty((v.shape[0], v.shape[1]), device=x.device)
        TargetIrrelated = torch.empty((v.shape[0], v.shape[1]), device=x.device)
        TargetRelatedi = torch.empty((vi.shape[0], vi.shape[1]), device=xi.device)
        TargetIrrelatedi = torch.empty((vi.shape[0], vi.shape[1]), device=xi.device)
        
        i = 0
        for bq, bk, bqi, bki, bmask in zip(q, k, qi, ki, mask):
            
            bq_TargetRelated = bq[bmask.unsqueeze(-1).expand(-1,768).bool()].view(-1,768)
            if in_dict["tokens_target_rgb"] is not None:
                bq_TargetRelated = torch.cat([bq_TargetRelated, in_dict["tokens_target_rgb"]], dim=0)
            # attn_TargetRelated = (bq_TargetRelated @ bk.transpose(-2, -1)) * self.scale
            # TargetRelated[i] = torch.exp(self.softplus(torch.mean(attn_TargetRelated, dim = 0)))
            
            bq_TargetIrrelated = bq[(1-bmask).unsqueeze(-1).expand(-1,768).bool()].view(-1,768)
            if in_dict["tokens_background_rgb"] is not None:
                bq_TargetIrrelated = torch.cat([bq_TargetRelated, in_dict["tokens_background_rgb"]], dim=0)
            # attn_TargetIrrelated = (bq_TargetIrrelated @ bk.transpose(-2, -1)) * self.scale
            # TargetIrrelated[i] = torch.exp(self.softplus(torch.mean(attn_TargetIrrelated, dim = 0)))
            
            
            bq_TargetRelatedi = bqi[bmask.unsqueeze(-1).expand(-1,768).bool()].view(-1,768)
            if in_dict["tokens_target_tir"] is not None:
                bq_TargetRelatedi = torch.cat([bq_TargetRelated, in_dict["tokens_target_tir"]], dim=0)
            # attn_TargetRelatedi = (bq_TargetRelatedi @ bki.transpose(-2, -1)) * self.scale
            # TargetRelatedi[i] = torch.exp(self.softplus(torch.mean(attn_TargetRelatedi, dim = 0)))
            
            bq_TargetIrrelatedi = bqi[(1-bmask).unsqueeze(-1).expand(-1,768).bool()].view(-1,768)
            if in_dict["tokens_background_tir"] is not None:
                bq_TargetIrrelatedi = torch.cat([bq_TargetRelated, in_dict["tokens_background_tir"]], dim=0)
            # attn_TargetIrrelatedi = (bq_TargetIrrelatedi @ bki.transpose(-2, -1)) * self.scale
            # TargetIrrelatedi[i] = torch.exp(self.softplus(torch.mean(attn_TargetIrrelatedi, dim = 0)))
            
            attn_TargetRelated = (torch.cat([bq_TargetRelated, bq_TargetRelatedi], dim=0) @ bk.transpose(-2, -1)) * self.scale
            TargetRelated[i] = torch.exp(self.softplus(torch.mean(attn_TargetRelated, dim = 0)))
            
            attn_TargetRelatedi = (torch.cat([bq_TargetRelated, bq_TargetRelatedi], dim=0) @ bki.transpose(-2, -1)) * self.scale
            TargetRelatedi[i] = torch.exp(self.softplus(torch.mean(attn_TargetRelatedi, dim = 0)))
            
            attn_TargetIrrelated = (torch.cat([bq_TargetIrrelated, bq_TargetIrrelatedi], dim=0) @ bk.transpose(-2, -1)) * self.scale
            TargetIrrelated[i] = torch.exp(self.softplus(torch.mean(attn_TargetIrrelated, dim = 0)))
            
            attn_TargetIrrelatedi = (torch.cat([bq_TargetIrrelated, bq_TargetIrrelatedi], dim=0) @ bki.transpose(-2, -1)) * self.scale
            TargetIrrelatedi[i] = torch.exp(self.softplus(torch.mean(attn_TargetIrrelatedi, dim = 0)))
            
            
            
            i += 1
        
        TargetRelated = self.Scale(TargetRelated, 2)
        TargetRelatedi = self.Scale(TargetRelatedi, 2)
        TargetIrrelated = self.Scale(TargetIrrelated)
        TargetIrrelatedi = self.Scale(TargetIrrelatedi)
        
        w = TargetRelatedi.mean(dim=1) / (TargetRelated.mean(dim=1) + TargetRelatedi.mean(dim=1))
        wi = TargetRelated.mean(dim=1) / (TargetRelated.mean(dim=1) + TargetRelatedi.mean(dim=1))
        
        W_TargetRelated = ((TargetRelated * w.unsqueeze(-1)) + (TargetRelatedi * wi.unsqueeze(-1)))
        W_TargetIrrelated = ((TargetIrrelated * w.unsqueeze(-1)) + (TargetIrrelatedi * wi.unsqueeze(-1)))
        
        W = W_TargetRelated - W_TargetIrrelated
        W_proj = self.proj_w(W)
        W_proj = self.Scale(W_proj, 2)
        
        search = search * W_proj.unsqueeze(-1)
        searchi = searchi * W_proj.unsqueeze(-1)
        
        # if in_dict["dynamic_template"] is None:
        #     x = torch.cat([template * (1 + mask.unsqueeze(-1)), search], dim=1)
        #     xi = torch.cat([templatei * (1 + mask.unsqueeze(-1)), searchi], dim=1)
        # else:
        #     x = torch.cat([template * (1 + mask[:, :len_t*2].unsqueeze(-1)), search, x[:, len_t*2+256:]], dim=1)
        #     xi = torch.cat([templatei * (1 + mask[:, :len_t*2].unsqueeze(-1)), searchi, xi[:, len_t*2+256:]], dim=1)
        x = torch.cat([template * (1 + mask.unsqueeze(-1)), search], dim=1)
        xi = torch.cat([templatei * (1 + mask.unsqueeze(-1)), searchi], dim=1)
        x = x*w.unsqueeze(1).unsqueeze(2) + xi*wi.unsqueeze(1).unsqueeze(2)
        x = x+self.proj(x)

        
        return x



################0.727################
# class MaskedCrossAttention(nn.Module):

#     def __init__(self, dim, attn_drop=0., proj_drop=0.):
#         super(MaskedCrossAttention, self).__init__()
#         # self.linear_q = nn.Linear(dim, dim)
#         # self.linear_k = nn.Linear(dim, dim)
#         # self.linear_v = nn.Linear(dim, dim)
#         # self.linear_qi = nn.Linear(dim, dim)
#         # self.linear_ki = nn.Linear(dim, dim)
#         # self.linear_vi = nn.Linear(dim, dim)
#         self.proj_wt = nn.Linear(256, 256)
#         self.proj_wb = nn.Linear(256, 256)
#         self.proj1 = nn.Linear(dim, dim)
#         # self.proj2 = nn.Linear(dim, dim)
#         self.dim = dim
#         self.scale = dim ** -0.5
#         self.attn_drop = nn.Dropout(attn_drop)
#         self.proj_drop = nn.Dropout(0.1)
#         self.softplus = nn.Softplus()
#         # self.kl = nn.KLDivLoss(reduction='batchmean')
    
#     def Scale(self, x, scale=1):
#         x_min,_ = x.min(dim=1)
#         x_max,_ = x.max(dim=1)
#         x_scaled = (x - x_min.unsqueeze(-1)) / (x_max.unsqueeze(-1) - x_min.unsqueeze(-1))
#         x_scaled = x_scaled * scale
#         return x_scaled
 
#     def forward(self, x, xi, mask, len_t):
        
#         mask = torch.cat([mask, mask], dim=1)
#         template = x[:, :len_t*2]
#         search = x[:, len_t*2:]
#         templatei = xi[:, :len_t*2]
#         searchi = xi[:, len_t*2:]
#         # q = self.linear_q(template)
#         # k = self.linear_k(search)
#         # v = self.linear_v(search)
#         # qi = self.linear_q(templatei)
#         # ki = self.linear_k(searchi)
#         # vi = self.linear_v(searchi)
#         q = template
#         k = search
#         v = search
#         qi = templatei
#         ki = searchi
#         vi = searchi
#         w_TargetRelated = torch.empty((v.shape[0], v.shape[1]), device=x.device)
#         w_TargetRelatedi = torch.empty((v.shape[0], v.shape[1]), device=xi.device)
#         w_TargetIrrelated = torch.empty((v.shape[0], v.shape[1]), device=x.device)
#         w_TargetIrrelatedi = torch.empty((v.shape[0], v.shape[1]), device=x.device)
#         i = 0
#         for b, k, bi, ki, mask in zip(q, k, qi, ki, mask):
            
#             q_TargetRelated = b[mask.unsqueeze(-1).expand(-1,q.shape[2]).bool()].view(-1,q.shape[-1])
#             attn_TargetRelated = (q_TargetRelated @ k.transpose(-2, -1)) * self.scale
#             attn_TargetRelated = self.attn_drop(attn_TargetRelated)
#             w_TargetRelated[i] = torch.exp(self.softplus(torch.mean(attn_TargetRelated, dim = 0)))
            
#             q_TargetIrrelated = b[(1-mask).unsqueeze(-1).expand(-1,q.shape[2]).bool()].view(-1,q.shape[-1])
#             attn_TargetIrrelated = (q_TargetIrrelated @ k.transpose(-2, -1)) * self.scale
#             attn_TargetIrrelated = self.attn_drop(attn_TargetIrrelated)
#             w_TargetIrrelated[i] = torch.exp(self.softplus(torch.mean(attn_TargetIrrelated, dim = 0)))
            
            
#             q_TargetRelatedi = bi[mask.unsqueeze(-1).expand(-1,qi.shape[2]).bool()].view(-1,qi.shape[-1])
#             attn_TargetRelatedi = (q_TargetRelatedi @ ki.transpose(-2, -1)) * self.scale
#             attn_TargetRelatedi = self.attn_drop(attn_TargetRelatedi)
#             w_TargetRelatedi[i] = torch.exp(self.softplus(torch.mean(attn_TargetRelatedi, dim = 0)))
            
#             q_TargetIrrelatedi = bi[(1-mask).unsqueeze(-1).expand(-1,qi.shape[2]).bool()].view(-1,qi.shape[-1])
#             attn_TargetIrrelatedi = (q_TargetIrrelatedi @ ki.transpose(-2, -1)) * self.scale
#             attn_TargetIrrelatedi = self.attn_drop(attn_TargetIrrelatedi)
#             w_TargetIrrelatedi[i] = torch.exp(self.softplus(torch.mean(attn_TargetIrrelatedi, dim = 0)))
            
#             i += 1
            
#         # fused_template = x[:, :len_t]
#         # a = (fused_template @ search.transpose(-2, -1)) * self.scale
#         # ai = (fused_template @ searchi.transpose(-2, -1)) * self.scale
#         # sub = a - ai
#         # sub = torch.where(sub > 0, torch.ones_like(sub), torch.zeros_like(sub))
#         # w = sub.sum(dim=1).sum(dim=1) / ((sub.sum(dim=1) + (1 - sub).sum(dim=1))).sum(dim=1)
#         # wi = (1 - sub).sum(dim=1).sum(dim=1) / ((sub.sum(dim=1) + (1 - sub).sum(dim=1))).sum(dim=1)
        
#         w_TargetRelated = self.Scale(w_TargetRelated, 2)
#         w_TargetRelatedi = self.Scale(w_TargetRelatedi, 2)
#         w_TargetIrrelated = self.Scale(w_TargetIrrelated)
#         w_TargetIrrelatedi = self.Scale(w_TargetIrrelatedi)
        
#         w = w_TargetRelatedi.mean(dim=1) / (w_TargetRelated.mean(dim=1) + w_TargetRelatedi.mean(dim=1))
#         wi = w_TargetRelated.mean(dim=1) / (w_TargetRelated.mean(dim=1) + w_TargetRelatedi.mean(dim=1))
        
#         W_TargetRelated = ((w_TargetRelated * w.unsqueeze(-1)) + (w_TargetRelatedi * wi.unsqueeze(-1)))
#         W_TargetIrrelated = ((w_TargetIrrelated * w.unsqueeze(-1)) + (w_TargetIrrelatedi * wi.unsqueeze(-1)))
        
#         W_TargetRelated = self.proj_wt(W_TargetRelated)
#         W_TargetIrrelated = self.proj_wb(W_TargetIrrelated)
#         # x_log = F.log_softmax(W_TargetRelated.reshape(v.shape[0],16,16),dim=1)
#         # y = F.softmax(W_TargetIrrelated.reshape(v.shape[0],16,16),dim=1)
#         # KL = self.kl(x_log, y) * 0.0005
#         W_TargetRelated = self.Scale(W_TargetRelated, 2)
#         W_TargetIrrelated = self.Scale(W_TargetIrrelated)

        
        
#         search = v * W.unsqueeze(-1) * W_proj.unsqueeze(-1)
#         searchi = vi * W.unsqueeze(-1) * W_proj.unsqueeze(-1)
        
#         x = torch.cat([template * (1 + mask.unsqueeze(-1)), search], dim=1)
#         # x = self.proj1(x)
#         # x = self.proj_drop(x)
#         xi = torch.cat([templatei * (1 + mask.unsqueeze(-1)), searchi], dim=1)
#         # xi = self.proj2(xi)
#         # xi = self.proj_drop(xi)
#         x = x*w.unsqueeze(1).unsqueeze(2) + xi*wi.unsqueeze(1).unsqueeze(2)
#         x = self.proj1(x)
#         x = self.proj_drop(x)
        
#         return x, x, 0


class Cross_Attention(nn.Module):
    def __init__(self, dim, mode, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.,
                 rpe=False, z_size=7, x_size=14):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5  # NOTE: Small scale for attention map normalization

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.mode = mode
        self.rpe =rpe
        if self.rpe:
            relative_position_index = \
                generate_2d_concatenated_self_attention_relative_positional_encoding_index([z_size, z_size],
                                                                                           [x_size, x_size])
            self.register_buffer("relative_position_index", relative_position_index)
            # define a parameter table of relative position bias
            self.relative_position_bias_table = nn.Parameter(torch.empty((num_heads,
                                                                          relative_position_index.max() + 1)))
            trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self, dom, ref, mask=None, return_attention=False):
        # x: B, N, C
        # mask: [B, N, ] torch.bool
        B, N, C = dom.shape
        
        lens_z = 64  # Number of template tokens
        lens_x = 256  # Number of search region tokens
        """
        elif self.mode=='t2t':  # Template to template
            q = x[:, :lens_z]  # B, lens_z, C
            k = x[:, lens_z:]  # B, lens_z, C
            v = x[:, lens_z:]  # B, lens_z, C
        elif self.mode=='s2s':  # Search to search
            q = x[:, :lens_x]  # B, lens_x, C
            k = x[:, lens_x:]  # B, lens_x, C
            v = x[:, lens_x:]  # B, lens_x, C
        """

        """
        q = dom[:, :lens_x]  # B, lens_x, C
        k = ref[:, lens_x:]  # B, lens_x, C
        v = ref[:, lens_x:]  # B, lens_x, C
        """

        q = dom
        k = ref
        v = ref

        attn = (q @ k.transpose(-2, -1)) * self.scale  # B, lens_z, lens_x; B, lens_x, lens_z
       
        if self.rpe:
            relative_position_bias = self.relative_position_bias_table[:, self.relative_position_index].unsqueeze(0)
            attn += relative_position_bias

        if mask is not None:
            attn = attn.masked_fill(mask.unsqueeze(1).unsqueeze(2), float('-inf'),)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = attn @ v  # B, lens_z/x, C

        x = x.transpose(1, 2)  # B, C, lens_z/x
        x = x.reshape(B, -1, C)  # B, lens_z/x, C; NOTE: Rearrange channels, marginal improvement
        x = self.proj(x)
        x = self.proj_drop(x)

        """
        if self.mode == 's2t':
            x = torch.cat([x, k], dim=1)
        elif self.mode == 't2s':
            x = torch.cat([k, x], dim=1)
        elif self.mode == 't2t':
            x = torch.cat([x, k], dim=1)
        elif self.mode == 's2s':
            x = torch.cat([x, k], dim=1)
        """

        if return_attention:
            return x, attn
        else:
            return x
        
        

    
    
# class Fusion(nn.Module):

#     def __init__(self, dim, proj_drop=0.):
#         super(Fusion, self).__init__()
#         # self.proj_w = nn.Linear(256, 256)
#         # self.proj = nn.Linear(dim, dim)
#         self.dim = dim
#         self.scale = dim ** -0.5
#         self.proj_drop = nn.Dropout(proj_drop)
#         self.softplus = nn.Softplus()
#         self.norm = nn.LayerNorm(dim)
#         # self.fc = nn.Sequential(
#         #     nn.Linear(dim*2, 1024),
#         #     nn.ReLU(),
#         #     nn.Linear(1024, dim),
#         #     nn.Sigmoid()
#         # )
#         # self.fc2 = nn.Sequential(
#         #     nn.Linear(dim, dim),
#         #     nn.Sigmoid()
#         # )
#         self.conv = nn.Conv1d(in_channels=dim*2, out_channels=dim, kernel_size=1)
#         self.proj_w = nn.Linear(256,256)
    
#     def Scale(self, x, scale=1):
#             x_min,_ = x.min(dim=1)
#             x_max,_ = x.max(dim=1)
#             x_scaled = (x - x_min.unsqueeze(-1)) / (x_max.unsqueeze(-1) - x_min.unsqueeze(-1))
#             x_scaled = x_scaled * scale
#             return x_scaled
 
#     def forward(self, x, xi, len_t, fused_template, mask):

#         search = x[:, len_t:]
#         template = x[:, :len_t]
#         searchi = xi[:, len_t:]
#         templatei = xi[:, :len_t]

#         q = template
#         k = search
#         v = search
#         qi = templatei
#         ki = searchi
#         vi = searchi
        
#         TargetRelated = torch.empty((v.shape[0], v.shape[1]), device=x.device)
#         # TargetIrrelated = torch.empty((v.shape[0], v.shape[1]), device=x.device)
#         TargetRelatedi = torch.empty((vi.shape[0], vi.shape[1]), device=xi.device)
#         # TargetIrrelatedi = torch.empty((vi.shape[0], vi.shape[1]), device=xi.device)
#         W_TargetRelated = torch.empty((v.shape[0], v.shape[1]), device=x.device)
        
#         search = (self.conv((torch.cat([search, searchi], dim=-1)).permute(0,2,1))).permute(0,2,1)
        
#         i = 0
#         for bq, bk, bqi, bki, bf, bs, bmask in zip(q, k, qi, ki,fused_template, search, mask):
            
#             bq_TargetRelated = bq[bmask.unsqueeze(-1).expand(-1,768).bool()].view(-1,768)
#             attn_TargetRelated = (bq_TargetRelated @ bk.transpose(-2, -1)) * self.scale
#             TargetRelated[i] = torch.exp(self.softplus(torch.mean(attn_TargetRelated, dim = 0)))
            
#             # bq_TargetIrrelated = bq[(1-bmask).unsqueeze(-1).expand(-1,768).bool()].view(-1,768)
#             # attn_TargetIrrelated = (bq_TargetIrrelated @ bk.transpose(-2, -1)) * self.scale
#             # TargetIrrelated[i] = torch.exp(self.softplus(torch.mean(attn_TargetIrrelated, dim = 0)))
            
            
#             bq_TargetRelatedi = bqi[bmask.unsqueeze(-1).expand(-1,768).bool()].view(-1,768)
#             attn_TargetRelatedi = (bq_TargetRelatedi @ bki.transpose(-2, -1)) * self.scale
#             TargetRelatedi[i] = torch.exp(self.softplus(torch.mean(attn_TargetRelatedi, dim = 0)))
            
#             # bq_TargetIrrelatedi = bqi[(1-bmask).unsqueeze(-1).expand(-1,768).bool()].view(-1,768)
#             # attn_TargetIrrelatedi = (bq_TargetIrrelatedi @ bki.transpose(-2, -1)) * self.scale
#             # TargetIrrelatedi[i] = torch.exp(self.softplus(torch.mean(attn_TargetIrrelatedi, dim = 0)))
            
#             bf_TargetRelated = bf[bmask.unsqueeze(-1).expand(-1,768).bool()].view(-1,768)
#             attn = (bf_TargetRelated @ bs.transpose(-2, -1)) * self.scale
#             W_TargetRelated[i] = torch.exp(self.softplus(torch.mean(attn, dim = 0)))
            
#             i += 1
        
#         TargetRelated = self.Scale(TargetRelated, 2)
#         TargetRelatedi = self.Scale(TargetRelatedi, 2)
#         # TargetIrrelated = self.Scale(TargetIrrelated)
#         # TargetIrrelatedi = self.Scale(TargetIrrelatedi)
#         W_TargetRelated = self.Scale(W_TargetRelated)
        
#         w_proj = self.proj_w(W_TargetRelated)
#         w_proj = self.Scale(w_proj)
        
#         w = TargetRelatedi.mean(dim=1) / (TargetRelated.mean(dim=1) + TargetRelatedi.mean(dim=1))
#         wi = TargetRelated.mean(dim=1) / (TargetRelated.mean(dim=1) + TargetRelatedi.mean(dim=1))
        
    
#         x = x*w.unsqueeze(1).unsqueeze(2) + xi*wi.unsqueeze(1).unsqueeze(2)
        
        
#         # search = self.fc(torch.cat([search, searchi], dim=-1))
#         # w_channel = self.fc2(search.mean(dim=1)).unsqueeze(1).expand_as(search)
        
        
#         # bq_TargetRelated = bq[bmask.unsqueeze(-1).expand(-1,768).bool()].view(-1,768)
#         # attn_TargetRelated = (fused_template @ search.transpose(-2, -1)) * self.scale
#         # W_TargetRelated = torch.exp(self.softplus(torch.mean(attn_TargetRelated, dim = 1)))
        
#         # W_TargetRelated = ((TargetRelated * w.unsqueeze(-1)) + (TargetRelatedi * wi.unsqueeze(-1)))
#         # w_proj = self.proj_w(W_TargetRelated)
        

#         search = search * (W_TargetRelated+w_proj).unsqueeze(-1) 
        
#         x = x + self.norm(torch.cat([fused_template, search], dim=1))
        
#         return x
    
    
    
    


# class Fusion(nn.Module):

#     def __init__(self, dim, proj_drop=0.):
#         super(Fusion, self).__init__()
#         # self.proj_w = nn.Linear(256, 256)
#         # self.proj = nn.Linear(dim, dim)
#         self.dim = dim
#         self.scale = dim ** -0.5
#         self.proj_drop = nn.Dropout(proj_drop)
#         self.softplus = nn.Softplus()
#         self.norm = nn.LayerNorm(dim)
#         # self.fc = nn.Sequential(
#         #     nn.Linear(dim*2, 1024),
#         #     nn.ReLU(),
#         #     nn.Linear(1024, dim),
#         #     nn.Sigmoid()
#         # )
#         # self.fc2 = nn.Sequential(
#         #     nn.Linear(dim, dim),
#         #     nn.Sigmoid()
#         # )
#         self.conv = nn.Conv1d(in_channels=dim*2, out_channels=dim, kernel_size=1)
#         # self.proj_w = nn.Linear(256,256)
#         self.fc2 = nn.Sequential(
#             nn.Linear(dim, dim),
#             nn.Sigmoid()
#         )
    
#     def Scale(self, x, scale=1):
#             x_min,_ = x.min(dim=1)
#             x_max,_ = x.max(dim=1)
#             x_scaled = (x - x_min.unsqueeze(-1)) / (x_max.unsqueeze(-1) - x_min.unsqueeze(-1))
#             x_scaled = x_scaled * scale
#             return x_scaled
 
#     def forward(self, x, xi, len_t, fused_template):

#         search = x[:, len_t:]
#         # template = x[:, :len_t]
#         searchi = xi[:, len_t:]
#         # templatei = xi[:, :len_t]

        
#         # fused_template = x[:, :len_t]
#         a = (fused_template @ search.transpose(-2, -1)) * self.scale
#         ai = (fused_template @ searchi.transpose(-2, -1)) * self.scale
#         sub = a - ai
#         sub = torch.where(sub > 0, torch.ones_like(sub), torch.zeros_like(sub))
#         w = sub.sum(dim=1).sum(dim=1) / ((sub.sum(dim=1) + (1 - sub).sum(dim=1))).sum(dim=1)
#         wi = (1 - sub).sum(dim=1).sum(dim=1) / ((sub.sum(dim=1) + (1 - sub).sum(dim=1))).sum(dim=1)
        
    
#         x = x*w.unsqueeze(1).unsqueeze(2) + xi*wi.unsqueeze(1).unsqueeze(2)
        
#         search = (self.conv((torch.cat([search, searchi], dim=-1)).permute(0,2,1))).permute(0,2,1)
#         # search = self.fc(torch.cat([search, searchi], dim=-1))
#         # w_channel = self.fc2(search.mean(dim=1)).unsqueeze(1).expand_as(search)
        
        
#         # bq_TargetRelated = bq[bmask.unsqueeze(-1).expand(-1,768).bool()].view(-1,768)
#         attn_TargetRelated = (fused_template @ search.transpose(-2, -1)) * self.scale
#         W_TargetRelated = torch.exp(self.softplus(torch.mean(attn_TargetRelated, dim = 1)))
        
#         # W_TargetRelated = ((TargetRelated * w.unsqueeze(-1)) + (TargetRelatedi * wi.unsqueeze(-1)))
#         # w_proj = self.proj_w(W_TargetRelated)
#         W_TargetRelated = self.Scale(W_TargetRelated)
        
#         w_channel = self.fc2(search.mean(dim=1)).unsqueeze(1).expand_as(search)

#         search = search * W_TargetRelated.unsqueeze(-1) * w_channel
        
#         x = x + self.norm(torch.cat([fused_template, search], dim=1))
        
#         return x


  
#0.730
class Fusion(nn.Module):

    def __init__(self, dim, proj_drop=0.):
        super(Fusion, self).__init__()
        # self.proj_w = nn.Linear(256, 256)
        # self.proj = nn.Linear(dim, dim)
        self.dim = dim
        self.scale = dim ** -0.5
        self.proj_drop = nn.Dropout(proj_drop)
        self.softplus = nn.Softplus()
        self.norm = nn.LayerNorm(dim)
        # self.fc = nn.Sequential(
        #     nn.Linear(dim*2, 1024),
        #     nn.ReLU(),
        #     nn.Linear(1024, dim),
        #     nn.Sigmoid()
        # )
        # self.fc2 = nn.Sequential(
        #     nn.Linear(dim, dim),
        #     nn.Sigmoid()
        # )
        self.conv = nn.Conv1d(in_channels=dim*2, out_channels=dim, kernel_size=1)
        # self.proj_w = nn.Linear(256,256)
    
    def Scale(self, x, scale=1):
            x_min,_ = x.min(dim=1)
            x_max,_ = x.max(dim=1)
            x_scaled = (x - x_min.unsqueeze(-1)) / (x_max.unsqueeze(-1) - x_min.unsqueeze(-1))
            x_scaled = x_scaled * scale
            return x_scaled
 
    def forward(self, x, xi, len_t, fused_template):

        search = x[:, len_t:]
        template = x[:, :len_t]
        searchi = xi[:, len_t:]
        # templatei = xi[:, :len_t]

        
        # fused_template = x[:, :len_t]
        a = (fused_template @ search.transpose(-2, -1)) * self.scale
        ai = (fused_template @ searchi.transpose(-2, -1)) * self.scale
        sub = a - ai
        sub = torch.where(sub > 0, torch.ones_like(sub), torch.zeros_like(sub))
        w = sub.sum(dim=1).sum(dim=1) / ((sub.sum(dim=1) + (1 - sub).sum(dim=1))).sum(dim=1)
        wi = (1 - sub).sum(dim=1).sum(dim=1) / ((sub.sum(dim=1) + (1 - sub).sum(dim=1))).sum(dim=1)
        
    
        x = x*w.unsqueeze(1).unsqueeze(2) + xi*wi.unsqueeze(1).unsqueeze(2)
        
        search = (self.conv((torch.cat([search, searchi], dim=-1)).permute(0,2,1))).permute(0,2,1)
        # search = self.fc(torch.cat([search, searchi], dim=-1))
        # w_channel = self.fc2(search.mean(dim=1)).unsqueeze(1).expand_as(search)
        
        
        # bq_TargetRelated = bq[bmask.unsqueeze(-1).expand(-1,768).bool()].view(-1,768)
        attn_TargetRelated = (fused_template @ search.transpose(-2, -1)) * self.scale
        W_TargetRelated = torch.exp(self.softplus(torch.mean(attn_TargetRelated, dim = 1)))
        
        # W_TargetRelated = ((TargetRelated * w.unsqueeze(-1)) + (TargetRelatedi * wi.unsqueeze(-1)))
        # w_proj = self.proj_w(W_TargetRelated)
        W_TargetRelated = self.Scale(W_TargetRelated)

        search = search * W_TargetRelated.unsqueeze(-1) 
        
        x = x + self.norm(torch.cat([template, search], dim=1))
        # x = x[:,64:,:]
        
        return x


# class Fusion(nn.Module):

#     def __init__(self, dim, proj_drop=0.):
#         super(Fusion, self).__init__()
#         # self.proj_w = nn.Linear(256, 256)
#         self.proj = nn.Linear(dim, dim)
#         self.dim = dim
#         self.scale = dim ** -0.5
#         self.proj_drop = nn.Dropout(proj_drop)
#         self.softplus = nn.Softplus()
#         self.norm = nn.LayerNorm(dim)
#         # self.fc = nn.Sequential(
#         #     nn.Linear(dim*2, 1024),
#         #     nn.ReLU(),
#         #     nn.Linear(1024, dim),
#         #     nn.Sigmoid()
#         # )
#         self.fc2 = nn.Sequential(
#             nn.Linear(dim, dim),
#             nn.Sigmoid()
#         )
#         self.conv = nn.Conv1d(in_channels=dim*2, out_channels=dim, kernel_size=1)
#         self.proj_w = nn.Linear(256,256)
    
#     def Scale(self, x, scale=1):
#             x_min,_ = x.min(dim=1)
#             x_max,_ = x.max(dim=1)
#             x_scaled = (x - x_min.unsqueeze(-1)) / (x_max.unsqueeze(-1) - x_min.unsqueeze(-1))
#             x_scaled = x_scaled * scale
#             return x_scaled
 
#     def forward(self, x, xi, len_t, fused_template, mask):

#         search = x[:, len_t:]
#         template = x[:, :len_t]
#         searchi = xi[:, len_t:]
#         templatei = xi[:, :len_t]

#         q = template
#         k = search
#         v = search
#         qi = templatei
#         ki = searchi
#         vi = searchi
        
#         TargetRelated = torch.empty((v.shape[0], v.shape[1]), device=x.device)
#         # TargetIrrelated = torch.empty((v.shape[0], v.shape[1]), device=x.device)
#         TargetRelatedi = torch.empty((vi.shape[0], vi.shape[1]), device=xi.device)
#         # TargetIrrelatedi = torch.empty((vi.shape[0], vi.shape[1]), device=xi.device)
#         W_TargetRelated = torch.empty((v.shape[0], v.shape[1]), device=x.device)
        
#         search = (self.conv((torch.cat([search, searchi], dim=-1)).permute(0,2,1))).permute(0,2,1)
        
#         i = 0
#         for bk, bki, bf, bs, bmask, bq, bqi in zip(k, ki,fused_template, search, mask, q, qi):
            
#             bf_TargetRelated = bf[bmask.unsqueeze(-1).expand(-1,768).bool()].view(-1,768)
#             attn = (bf_TargetRelated @ bs.transpose(-2, -1)) * self.scale
#             W_TargetRelated[i] = torch.exp(self.softplus(torch.mean(attn, dim = 0)))
            
#             bq_TargetRelated = bq[bmask.unsqueeze(-1).expand(-1,768).bool()].view(-1,768)
#             attn_TargetRelated = (bq_TargetRelated @ bk.transpose(-2, -1)) * self.scale
#             TargetRelated[i] = torch.exp(self.softplus(torch.mean(attn_TargetRelated, dim = 0)))
            
#             # bq_TargetIrrelated = bq[(1-bmask).unsqueeze(-1).expand(-1,768).bool()].view(-1,768)
#             # attn_TargetIrrelated = (bq_TargetIrrelated @ bk.transpose(-2, -1)) * self.scale
#             # TargetIrrelated[i] = torch.exp(self.softplus(torch.mean(attn_TargetIrrelated, dim = 0)))
            
            
#             bq_TargetRelatedi = bqi[bmask.unsqueeze(-1).expand(-1,768).bool()].view(-1,768)
#             attn_TargetRelatedi = (bq_TargetRelatedi @ bki.transpose(-2, -1)) * self.scale
#             TargetRelatedi[i] = torch.exp(self.softplus(torch.mean(attn_TargetRelatedi, dim = 0)))
            
#             # bq_TargetIrrelatedi = bqi[(1-bmask).unsqueeze(-1).expand(-1,768).bool()].view(-1,768)
#             # attn_TargetIrrelatedi = (bq_TargetIrrelatedi @ bki.transpose(-2, -1)) * self.scale
#             # TargetIrrelatedi[i] = torch.exp(self.softplus(torch.mean(attn_TargetIrrelatedi, dim = 0)))
            

            
#             i += 1
        
#         TargetRelated = self.Scale(TargetRelated, 2)
#         TargetRelatedi = self.Scale(TargetRelatedi, 2)
#         # TargetIrrelated = self.Scale(TargetIrrelated)
#         # TargetIrrelatedi = self.Scale(TargetIrrelatedi)
#         W_TargetRelated = self.Scale(W_TargetRelated)
        
#         w_proj = self.proj_w(W_TargetRelated)
#         w_proj = self.Scale(w_proj)
        
#         w = TargetRelatedi.mean(dim=1) / (TargetRelated.mean(dim=1) + TargetRelatedi.mean(dim=1))
#         wi = TargetRelated.mean(dim=1) / (TargetRelated.mean(dim=1) + TargetRelatedi.mean(dim=1))
        
    
#         x = x*w.unsqueeze(1).unsqueeze(2) + xi*wi.unsqueeze(1).unsqueeze(2)
        
        
#         # search = self.fc(torch.cat([search, searchi], dim=-1))
#         w_channel = self.fc2(search.mean(dim=1)).unsqueeze(1).expand_as(search)
        
        
#         # bq_TargetRelated = bq[bmask.unsqueeze(-1).expand(-1,768).bool()].view(-1,768)
#         # attn_TargetRelated = (fused_template @ search.transpose(-2, -1)) * self.scale
#         # W_TargetRelated = torch.exp(self.softplus(torch.mean(attn_TargetRelated, dim = 1)))
        
#         W_TargetRelated = ((TargetRelated * w.unsqueeze(-1)) + (TargetRelatedi * wi.unsqueeze(-1)))
#         w_proj = self.proj_w(W_TargetRelated)
        

#         search = search * (W_TargetRelated+w_proj).unsqueeze(-1) + w_channel * search
#         search = search + self.proj(search)
        
#         x = x + self.norm(torch.cat([fused_template, search], dim=1))
        
#         return x
    