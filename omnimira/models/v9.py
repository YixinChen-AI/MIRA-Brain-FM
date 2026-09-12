"""OmniMIRA v9 inference architecture."""
from __future__ import annotations
import torch
from torch import nn
from torch.nn import functional as F

class CondLN(nn.Module):
    def __init__(self, dim=128, modalities=4):
        super().__init__(); self.norm=nn.LayerNorm(dim,elementwise_affine=False); self.scale=nn.Embedding(modalities,dim); self.shift=nn.Embedding(modalities,dim); nn.init.zeros_(self.scale.weight); nn.init.zeros_(self.shift.weight)
    def forward(self,x,modality): return self.norm(x)*(1+self.scale(modality)[:,None])+self.shift(modality)[:,None]

class Block(nn.Module):
    def __init__(self,dim=128,modalities=4,ratio=4):
        super().__init__(); self.norm=CondLN(dim,modalities); self.mlp=nn.Sequential(nn.Linear(dim,dim*ratio),nn.GELU(),nn.Linear(dim*ratio,dim))
    def forward(self,x,modality): return x+self.mlp(self.norm(x,modality))

def position_encoding(grid,dim):
    per_axis=(dim//3)//2*2; coordinates=torch.meshgrid(*[torch.arange(g) for g in grid],indexing="ij"); frequency=1/(10000**(torch.arange(per_axis//2).float()/(per_axis/2))); parts=[]
    for coordinate in coordinates:
        angle=coordinate.reshape(-1,1).float()*frequency; parts.extend([angle.sin(),angle.cos()])
    return F.pad(torch.cat(parts,-1),(0,dim-per_axis*3))[None]

class AART(nn.Module):
    def __init__(self,dim=128,components=4):
        super().__init__(); self.queries=nn.Parameter(torch.randn(components,dim)*0.02); self.proj=nn.Linear(components*dim,dim)
    def forward(self,patches,membership):
        if membership.dtype!=torch.bool or membership.ndim!=2: raise ValueError("membership must be bool [ROI, patch]")
        rows=[]; valid=membership.any(-1)
        for mask in membership:
            if mask.any():
                selected=patches[:,mask]; score=torch.einsum("kd,bnd->bkn",self.queries,selected); pooled=torch.einsum("bkn,bnd->bkd",score.softmax(-1),selected); rows.append(self.proj(pooled.flatten(1)))
            else: rows.append(patches.new_zeros(patches.shape[0],patches.shape[-1]))
        return torch.stack(rows,1),valid[None].expand(patches.shape[0],-1)

class OmniMIRAV9(nn.Module):
    def __init__(self,img_size=(96,112,96),patch_size=8,dim=128,depth=4,components=4,modalities=4,contrastive_dim=64):
        super().__init__(); self.img_size=tuple(img_size); self.patch_size=patch_size; self.modalities=modalities
        self.patch_embed=nn.ModuleList([nn.Conv3d(1,dim,patch_size,stride=patch_size) for _ in range(modalities)]); self.blocks=nn.ModuleList([Block(dim,modalities) for _ in range(depth)]); self.norm=CondLN(dim,modalities)
        self.register_buffer("position",position_encoding([s//patch_size for s in img_size],dim),persistent=False); self.aart=AART(dim,components); self.patch_decoder=nn.Linear(dim,patch_size**3); self.contrast=nn.Linear(dim,contrastive_dim); self.volume=nn.Linear(dim,1)
    def encode(self,x,modality,memberships):
        if x.shape[1:]!=(1,*self.img_size): raise ValueError("Input shape does not match the OmniMIRA model grid")
        if modality.shape!=(x.shape[0],) or modality.dtype!=torch.long: raise ValueError("Invalid modality IDs")
        patches=self.position.new_zeros(x.shape[0],self.position.shape[1],self.position.shape[2])
        for mid,embedding in enumerate(self.patch_embed):
            selected=modality==mid
            if selected.any(): patches[selected]=embedding(x[selected]).flatten(2).transpose(1,2)
        patches=patches+self.position
        for block in self.blocks: patches=block(patches,modality)
        patches=self.norm(patches,modality); atlases={}
        for name,membership in memberships.items():
            tokens,valid=self.aart(patches,membership); atlases[name]={"tokens":tokens,"valid":valid}
        return {"patches":patches,"atlases":atlases}
    def forward(self,x,modality,memberships): return self.encode(x,modality,memberships)
