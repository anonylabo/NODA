import torch.nn as nn
import math
import torch


class Temporal_SelfAttention(nn.Module):
    def __init__(self, d_model, n_head, len):
        super(Temporal_SelfAttention, self).__init__()

        self.query_projection = nn.Linear(d_model, d_model, bias=False)
        self.key_projection = nn.Linear(d_model, d_model, bias=False)
        self.value_projection = nn.Linear(d_model, d_model, bias=False)

        self.e_projection = nn.Linear(d_model//n_head, len, bias=False)

        self.out_projection = nn.Linear(d_model, d_model)
        self.n_head = n_head
        self.len = len

    def forward(self, x, key_indices):
        B, L, _ = x.shape
        H = self.n_head

        queries = self.query_projection(x).view(B, L, H, -1)
        keys = self.key_projection(x).view(B, L, H, -1)
        values = self.value_projection(x).view(B, L, H, -1)

        e = self.e_projection(queries).permute(0,2,1,3)

        m = nn.ReflectionPad2d((0,self.len-1,0,0))
        e = nn.functional.pad(m(e), (0,1,0,self.len-1))
        e = e.reshape(B, H, e.shape[-1], e.shape[-2])
        s_rel = e[:,:,:self.len,self.len-1:]

        scale = 1. / math.sqrt(queries.shape[-1])

        scores = torch.einsum("blhd,bshd->bhls", queries, keys)

        A = torch.softmax(scale * (scores * s_rel), dim=-1)
        V = torch.einsum("bhls,bshd->blhd", A, values)
        out = V.contiguous()

        out = out.view(B, L, -1)

        return self.out_projection(out), A

class Spatial_SelfAttention(nn.Module):
    def __init__(self, d_model, n_head):
        super(Spatial_SelfAttention, self).__init__()

        self.query_projection = nn.Linear(d_model, d_model, bias=False)
        self.key_projection = nn.Linear(d_model, d_model, bias=False)
        self.value_projection = nn.Linear(d_model, d_model, bias=False)
        self.out_projection = nn.Linear(d_model, d_model)
        self.n_head = n_head

    def forward(self, x, key_indices):
        B, L, _ = x.shape
        H = self.n_head

        queries = self.query_projection(x).view(B, L, H, -1).permute(0,2,1,3)
        keys = self.key_projection(x).view(B, L, H, -1).permute(0,2,1,3)
        values = self.value_projection(x).view(B, L, H, -1).permute(0,2,1,3)

        scale = 1. / math.sqrt(queries.shape[-1])

        keys_sample = keys[:, :, key_indices, :]
        values_sample = values[:, :, key_indices, :]

        scores = torch.einsum('bhlkd,bhlds->bhlks', queries.unsqueeze(-2), keys_sample.transpose(-2,-1))

        A = torch.softmax(scale * scores, dim=-1)
        V = torch.einsum('bhlks,bhlsd->bhlkd', A, values_sample).squeeze(dim=3)
        out = V.permute(0,2,1,3).contiguous()

        out = out.view(B, L, -1)

        return self.out_projection(out), A.squeeze()