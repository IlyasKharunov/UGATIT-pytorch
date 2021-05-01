import torch
import torch.nn.functional as F
from torch import nn, einsum

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# helpers

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def divisible_by(val, divisor):
    return (val % divisor) == 0

def unfold_output_size(image_size, kernel_size, stride, padding):
    return int(((image_size - kernel_size + (2 * padding)) / stride) + 1)

# classes

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, mult = 4, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim)
        )

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(
        self,
        *,
        dim,
        heads = 8,
        head_dim = 64,
        dropout = 0.
    ):
        super().__init__()
        inner_dim = heads * head_dim
        self.heads =  heads
        self.scale = head_dim ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        b, n, d, h = *x.shape, self.heads
        q, k, v = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h = h), (q, k, v))

        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale
        attn = sim.softmax(dim = -1)

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h = h)
        return self.to_out(out)
    
class Enc_Dec_Attention(nn.Module):
    def __init__(
        self,
        *,
        dim,
        inner_dim = 320,
        dropout = 0.
    ):
        super().__init__()
        self.scale = inner_dim ** -0.5
        self.to_q = nn.Linear(dim, inner_dim, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, k, v):
        b, n, d = *x.shape,
        q = self.to_q(x)
    
        #print(q.size(), k.size(), v.size())
        sim = einsum('b i d, b d -> b i', q, k) * self.scale
        attn = sim.sigmoid()

        out = einsum('b i, b d -> b i d', attn, v)
        return self.to_out(out)

# to_kv = PreNorm(nn.Linear(dim, inner_dim * 2, bias = False))
# k,v = to_kv(x).chunk(2, dim = -1)

# main class

class TNT(nn.Module):
    def __init__(
        self,
        *,
        image_size,
        patch_dim,
        pixel_dim,
        patch_size,
        pixel_size,
        enc_depth,
        dec_depth,
        num_channels,
        heads = 8,
        head_dim = 64,
        ff_dropout = 0.,
        attn_dropout = 0.,
        unfold_args = None
    ):
        super().__init__()
        assert divisible_by(image_size, patch_size), 'image size must be divisible by patch size'
        assert divisible_by(patch_size, pixel_size), 'patch size must be divisible by pixel size for now'

        im_h = im_w = image_size // patch_size
        num_patch_tokens = im_h*im_w
        enc_dec_dim = head_dim * 5
        self.image_size = image_size
        self.patch_size = patch_size
        self.patch_tokens = nn.Parameter(torch.randn(num_patch_tokens, patch_dim))
        # +1

        unfold_args = default(unfold_args, (pixel_size, pixel_size, 0))
        unfold_args = (*unfold_args, 0) if len(unfold_args) == 2 else unfold_args
        kernel_size, stride, padding = unfold_args

        pixel_width = unfold_output_size(patch_size, kernel_size, stride, padding)
        num_pixels = pixel_width ** 2

        self.to_pixel_tokens = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> (b h w) (p1 p2) c', p1 = patch_size, p2 = patch_size)
        )
        
        self.to_image = nn.Sequential(
            Rearrange('(b h w) (k1 k2) c -> b c (h k1) (w k2)', h = im_h, w = im_w, k1 = pixel_width, k2 = pixel_width)
        )

        self.patch_pos_emb = nn.Parameter(torch.randn(num_patch_tokens + 1, patch_dim))
        self.pixel_pos_emb = nn.Parameter(torch.randn(num_pixels, pixel_dim))

        enc_layers = nn.ModuleList([])
        for _ in range(enc_depth):

            pixel_to_patch = nn.Sequential(
                nn.LayerNorm(pixel_dim),
                Rearrange('... n d -> ... (n d)'),
                nn.Linear(pixel_dim * num_pixels, patch_dim),
            )

            enc_layers.append(nn.ModuleList([
                PreNorm(pixel_dim, Attention(dim = pixel_dim, heads = heads, head_dim = head_dim, dropout = attn_dropout)),
                PreNorm(pixel_dim, FeedForward(dim = pixel_dim, dropout = ff_dropout)),
                pixel_to_patch,
                PreNorm(patch_dim, Attention(dim = patch_dim, heads = heads, head_dim = head_dim, dropout = attn_dropout)),
                PreNorm(patch_dim, FeedForward(dim = patch_dim, dropout = ff_dropout)),
            ]))
        
        self.rearr = Rearrange('b n d -> (b n) d')
        self.to_kv = PreNorm(patch_dim, nn.Linear(patch_dim, enc_dec_dim * 2, bias = False))
        
        dec_layers = nn.ModuleList([])
        for _ in range(dec_depth):
            dec_layers.append(nn.ModuleList([
                PreNorm(pixel_dim, Attention(dim = pixel_dim, heads = heads, head_dim = head_dim, dropout = attn_dropout)),
                PreNorm(pixel_dim, Enc_Dec_Attention(dim = pixel_dim, inner_dim = enc_dec_dim, dropout = attn_dropout)),
                PreNorm(pixel_dim, FeedForward(dim = pixel_dim, dropout = ff_dropout))
            ]))
            
        self.enc_layers = enc_layers
        self.dec_layers = dec_layers

    def forward(self, x):
        b, _, h, w, patch_size, image_size = *x.shape, self.patch_size, self.image_size
        assert divisible_by(h, patch_size) and divisible_by(w, patch_size), f'height {h} and width {w} of input must be divisible by the patch size'

        num_patches_h = h // patch_size
        num_patches_w = w // patch_size
        n = num_patches_w * num_patches_h
        
        pixels = self.to_pixel_tokens(x)
        #print(pixels.size())
        patches = repeat(self.patch_tokens[:n], 'n d -> b n d', b = b)

        patches += rearrange(self.patch_pos_emb[:n], 'n d -> () n d')
        #n+1
        pixels += rearrange(self.pixel_pos_emb, 'n d -> () n d')

        for pixel_attn, pixel_ff, pixel_to_patch_residual, patch_attn, patch_ff in self.enc_layers:

            pixels = pixel_attn(pixels) + pixels
            pixels = pixel_ff(pixels) + pixels

            patches_residual = pixel_to_patch_residual(pixels)

            patches_residual = rearrange(patches_residual, '(b h w) d -> b (h w) d', h = num_patches_h, w = num_patches_w)
            #patches_residual = F.pad(patches_residual, (0, 0, 1, 0), value = 0) # cls token gets residual of 0
            patches = patches + patches_residual

            patches = patch_attn(patches) + patches
            patches = patch_ff(patches) + patches
        
        #print(patches.size(), pixels.size())
        patches = self.rearr(patches)
        patch_k, patch_v = self.to_kv(patches).chunk(2, dim = -1)
        #print(patch_k.size())

        for pixel_attn, enc_dec_attn, ff in self.dec_layers:

            pixels = pixel_attn(pixels) + pixels
            pixels = enc_dec_attn(pixels, k = patch_k, v = patch_v) + pixels
            #print(pixels.size())
            pixels = ff(pixels) + pixels
            #print(pixels.size())
        imag = self.to_image(pixels)

        return imag
