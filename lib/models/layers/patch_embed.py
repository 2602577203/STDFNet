import torch.nn as nn
import torch

from timm.models.layers import to_2tuple


class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, norm_layer=None, flatten=True):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        # allow different input size
        # B, C, H, W = x.shape
        # _assert(H == self.img_size[0], f"Input image height ({H}) doesn't match model ({self.img_size[0]}).")
        # _assert(W == self.img_size[1], f"Input image width ({W}) doesn't match model ({self.img_size[1]}).")
        #print("start",x.size())    #start torch.Size([1, 3, 256, 256])
        x = self.proj(x)           #flatten before torch.Size([1, 768, 16, 16])
        if self.flatten:
            #print("flatten before",x.size())       #flatten before torch.Size([1, 768, 16, 16])
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
            #print("flatten transpose",x.size())   #flatten transpose torch.Size([1, 256, 768])
        x = self.norm(x)
        #print("after",x.size())                #after torch.Size([1, 256, 768])
        return x

class SE_PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, ratio=16, norm_layer=None, flatten=True):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()
        
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            # nn.Linear(embed_dim, embed_dim // ratio, bias=False),  # 从 c -> c/r
            # nn.ReLU(),
            nn.Linear(embed_dim, embed_dim, bias=False),  # 从 c/r -> c
            nn.Sigmoid()
        )
        

    def forward(self, x):
        # allow different input size
        # B, C, H, W = x.shape
        # _assert(H == self.img_size[0], f"Input image height ({H}) doesn't match model ({self.img_size[0]}).")
        # _assert(W == self.img_size[1], f"Input image width ({W}) doesn't match model ({self.img_size[1]}).")
        #print("start",x.size())    #start torch.Size([1, 3, 256, 256])
        x = self.proj(x)           #flatten before torch.Size([1, 768, 16, 16])
        
        b, c, h, w = x.size()
        # Fsq操作：经池化后输出b*c的矩阵
        y = self.gap(x).view(b, c)
        # Fex操作：经全连接层输出（b，c，1，1）矩阵
        y = self.fc(y).view(b, c, 1, 1)
        x = x * y.expand_as(x)
        
        if self.flatten:
            #print("flatten before",x.size())       #flatten before torch.Size([1, 768, 16, 16])
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
            #print("flatten transpose",x.size())   #flatten transpose torch.Size([1, 256, 768])
        x = self.norm(x)
        #print("after",x.size())                #after torch.Size([1, 256, 768])
        return x
    
class SE_fusion(nn.Module):
    """ 2D Image to Patch Embedding
    """

    def __init__(self, embed_dim=768, lens_t=64, flatten=True):
        super().__init__()

        self.norm = nn.LayerNorm(embed_dim)
        self.flatten = flatten
        self.lens_t = lens_t
        self.fc1 = nn.Sequential(
            nn.Linear(embed_dim*2, 1024),
            nn.ReLU(),
            nn.Linear(1024, embed_dim),
            nn.Sigmoid()
        )
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            # nn.Linear(embed_dim, embed_dim // ratio, bias=False),  # 从 c -> c/r
            # nn.ReLU(),
            nn.Linear(embed_dim, embed_dim, bias=False),  # 从 c/r -> c
            nn.Sigmoid()
        )
        
    def SE(self, x):
        h = int(x.shape[1] ** 0.5)
        x = x.permute(0, 2, 1)
        x = x.view(32, 768, h, h)
        b, c, h, w = x.size()
        # Fsq操作：经池化后输出b*c的矩阵
        y = self.gap(x).view(b, c)
        # Fex操作：经全连接层输出（b，c，1，1）矩阵
        y = self.fc(y).view(b, c, 1, 1)
        x = x * y.expand_as(x)
        if self.flatten:
            #print("flatten before",x.size())       #flatten before torch.Size([1, 768, 16, 16])
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        return x
        
    def forward(self, x, xi):
        x = torch.cat([x, xi], dim=-1)
        x = self.fc1(x)
        template,search = x[:, self.lens_t:,:], x[:, :self.lens_t,:]
        template = self.SE(template)
        search = self.SE(search)
        x = torch.cat([template, search], dim=1)
        x = self.norm(x)
        return x
