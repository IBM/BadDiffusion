import math
from inspect import isfunction
from functools import partial
from sched import scheduler

# %matplotlib inline
from einops import rearrange
import numpy as np

import torch
from torch import nn, einsum
from torch.optim import Adam

from dataset import DatasetLoader

# """## What is a diffusion model?

# A (denoising) diffusion model isn't that complex if you compare it to other generative models such as Normalizing Flows, GANs or VAEs: they all convert noise from some simple distribution to a data sample. This is also the case here where **a neural network learns to gradually denoise data** starting from pure noise. 

# In a bit more detail for images, the set-up consists of 2 processes:
# * a fixed (or predefined) forward diffusion process $q$ of our choosing, that gradually adds Gaussian noise to an image, until you end up with pure noise
# * a learned reverse denoising diffusion process $p_\theta$, where a neural network is trained to gradually denoise an image starting from pure noise, until you end up with an actual image.

# <p align="center">
#     <img src="https://drive.google.com/uc?id=1t5dUyJwgy2ZpDAqHXw7GhUAp2FE5BWHA" width="600" />
# </p>

# Both the forward and reverse process indexed by \\(t\\) happen for some number of finite time steps \\(T\\) (the DDPM authors use \\(T=1000\\)). You start with \\(t=0\\) where you sample a real image \\(\mathbf{x}_0\\) from your data distribution (let's say an image of a cat from ImageNet), and the forward process samples some noise from a Gaussian distribution at each time step \\(t\\), which is added to the image of the previous time step. Given a sufficiently large \\(T\\) and a well behaved schedule for adding noise at each time step, you end up with what is called an [isotropic Gaussian distribution](https://math.stackexchange.com/questions/1991961/gaussian-distribution-is-isotropic) at \\(t=T\\) via a gradual process.

# ## In more mathematical form

# Let's write this down more formally, as ultimately we need a tractable loss function which our neural network needs to optimize. 

# Let \\(q(\mathbf{x}_0)\\) be the real data distribution, say of "real images". We can sample from this distribution to get an image, \\(\mathbf{x}_0 \sim q(\mathbf{x}_0)\\). We define the forward diffusion process \\(q(\mathbf{x}_t | \mathbf{x}_{t-1})\\) which adds Gaussian noise at each time step \\(t\\), according to a known variance schedule \\(0 < \beta_1 < \beta_2 < ... < \beta_T < 1\\) as
# $$
# q(\mathbf{x}_t | \mathbf{x}_{t-1}) = \mathcal{N}(\mathbf{x}_t; \sqrt{1 - \beta_t} \mathbf{x}_{t-1}, \beta_t \mathbf{I}). 
# $$

# Recall that a normal distribution (also called Gaussian distribution) is defined by 2 parameters: a mean \\(\mu\\) and a variance \\(\sigma^2 \geq 0\\). Basically, each new (slightly noiser) image at time step \\(t\\) is drawn from a **conditional Gaussian distribution** with \\(\mathbf{\mu}_t = \sqrt{1 - \beta_t} \mathbf{x}_{t-1}\\) and \\(\sigma^2_t = \beta_t\\), which we can do by sampling \\(\mathbf{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})\\) and then setting \\(\mathbf{x}_t = \sqrt{1 - \beta_t} \mathbf{x}_{t-1} +  \sqrt{\beta_t} \mathbf{\epsilon}\\). 

# Note that the \\(\beta_t\\) aren't constant at each time step \\(t\\) (hence the subscript) --- in fact one defines a so-called **"variance schedule"**, which can be linear, quadratic, cosine, etc. as we will see further (a bit like a learning rate schedule). 

# So starting from \\(\mathbf{x}_0\\), we end up with \\(\mathbf{x}_1,  ..., \mathbf{x}_t, ..., \mathbf{x}_T\\), where \\(\mathbf{x}_T\\) is pure Gaussian noise if we set the schedule appropriately.

# Now, if we knew the conditional distribution \\(p(\mathbf{x}_{t-1} | \mathbf{x}_t)\\), then we could run the process in reverse: by sampling some random Gaussian noise \\(\mathbf{x}_T\\), and then gradually "denoise" it so that we end up with a sample from the real distribution \\(\mathbf{x}_0\\).

# However, we don't know \\(p(\mathbf{x}_{t-1} | \mathbf{x}_t)\\). It's intractable since it requires knowing the distribution of all possible images in order to calculate this conditional probability. Hence, we're going to leverage a neural network to **approximate (learn) this conditional probability distribution**, let's call it \\(p_\theta (\mathbf{x}_{t-1} | \mathbf{x}_t)\\), with \\(\theta\\) being the parameters of the neural network, updated by gradient descent. 

# Ok, so we need a neural network to represent a (conditional) probability distribution of the backward process. If we assume this reverse process is Gaussian as well, then recall that any Gaussian distribution is defined by 2 parameters:
# * a mean parametrized by \\(\mu_\theta\\);
# * a variance parametrized by \\(\Sigma_\theta\\);

# so we can parametrize the process as 
# $$ p_\theta (\mathbf{x}_{t-1} | \mathbf{x}_t) = \mathcal{N}(\mathbf{x}_{t-1}; \mu_\theta(\mathbf{x}_{t},t), \Sigma_\theta (\mathbf{x}_{t},t))$$
# where the mean and variance are also conditioned on the noise level \\(t\\).

# Hence, our neural network needs to learn/represent the mean and variance. However, the DDPM authors decided to **keep the variance fixed, and let the neural network only learn (represent) the mean \\(\mu_\theta\\) of this conditional probability distribution**. From the paper:

# > First, we set \\(\Sigma_\theta ( \mathbf{x}_t, t) = \sigma^2_t \mathbf{I}\\) to untrained time dependent constants. Experimentally, both \\(\sigma^2_t = \beta_t\\) and \\(\sigma^2_t  = \tilde{\beta}_t\\) (see paper) had similar results. 

# This was then later improved in the [Improved diffusion models](https://openreview.net/pdf?id=-NEXDKk8gZ) paper, where a neural network also learns the variance of this backwards process, besides the mean.

# So we continue, assuming that our neural network only needs to learn/represent the mean of this conditional probability distribution.

# ## Defining an objective function (by reparametrizing the mean)

# To derive an objective function to learn the mean of the backward process, the authors observe that the combination of \\(q\\) and \\(p_\theta\\) can be seen as a variational auto-encoder (VAE) [(Kingma et al., 2013)](https://arxiv.org/abs/1312.6114). Hence, the **variational lower bound** (also called ELBO) can be used to minimize the negative log-likelihood with respect to ground truth data sample \\(\mathbf{x}_0\\) (we refer to the VAE paper for details regarding ELBO). It turns out that the ELBO for this process is a sum of losses at each time step \\(t\\), \\(L = L_0 + L_1 + ... + L_T\\). By construction of the forward \\(q\\) process and backward process, each term (except for \\(L_0\\)) of the loss is actually the **KL divergence between 2 Gaussian distributions** which can be written explicitly as an L2-loss with respect to the means!

# A direct consequence of the constructed forward process \\(q\\), as shown by Sohl-Dickstein et al., is that we can sample \\(\mathbf{x}_t\\) at any arbitrary noise level conditioned on \\(\mathbf{x}_0\\) (since sums of Gaussians is also Gaussian). This is very convenient:  we don't need to apply \\(q\\) repeatedly in order to sample \\(\mathbf{x}_t\\). 
# We have that 
# $$q(\mathbf{x}_t | \mathbf{x}_0) = \cal{N}(\mathbf{x}_t; \sqrt{\bar{\alpha}_t} \mathbf{x}_0, (1- \bar{\alpha}_t) \mathbf{I})$$

# with \\(\alpha_t := 1 - \beta_t\\) and \\(\bar{\alpha}t := \Pi_{s=1}^{t} \alpha_s\\). Let's refer to this equation as the "nice property". This means we can sample Gaussian noise and scale it appropriatly and add it to \\(\mathbf{x}_0\\) to get \\(\mathbf{x}_t\\) directly. Note that the \\(\bar{\alpha}_t\\) are functions of the known \\(\beta_t\\) variance schedule and thus are also known and can be precomputed. This then allows us, during training, to **optimize random terms of the loss function \\(L\\)** (or in other words, to randomly sample \\(t\\) during training and optimize \\(L_t\\).

# Another beauty of this property, as shown in Ho et al. is that one can (after some math, for which we refer the reader to [this excellent blog post](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/)) instead **reparametrize the mean to make the neural network learn (predict) the added noise (via a network \\(\mathbf{\epsilon}_\theta(\mathbf{x}_t, t)\\) for noise level \\(t\\)** in the KL terms which constitute the losses. This means that our neural network becomes a noise predictor, rather than a (direct) mean predictor. The mean can be computed as follows:

# $$ \mathbf{\mu}_\theta(\mathbf{x}_t, t) = \frac{1}{\sqrt{\alpha_t}} \left(  \mathbf{x}_t - \frac{\beta_t}{\sqrt{1- \bar{\alpha}_t}} \mathbf{\epsilon}_\theta(\mathbf{x}_t, t) \right)$$

# The final objective function \\(L_t\\) then looks as follows (for a random time step \\(t\\) given \\(\mathbf{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})\\) ): 

# $$ \| \mathbf{\epsilon} - \mathbf{\epsilon}_\theta(\mathbf{x}_t, t) \|^2 = \| \mathbf{\epsilon} - \mathbf{\epsilon}_\theta( \sqrt{\bar{\alpha}_t} \mathbf{x}_0 + \sqrt{(1- \bar{\alpha}_t)  } \mathbf{\epsilon}, t) \|^2.$$

# Here, \\(\mathbf{x}_0\\) is the initial (real, uncorruped) image, and we see the direct noise level \\(t\\) sample given by the fixed forward process. \\(\mathbf{\epsilon}\\) is the pure noise sampled at time step \\(t\\), and \\(\mathbf{\epsilon}_\theta (\mathbf{x}_t, t)\\) is our neural network. The neural network is optimized using a simple mean squared error (MSE) between the true and the predicted Gaussian noise.

# The training algorithm now looks as follows:


# <p align="center">
#     <img src="https://drive.google.com/uc?id=1LJsdkZ3i1J32lmi9ONMqKFg5LMtpSfT4" width="400" />
# </p>

# In other words:
# * we take a random sample $\mathbf{x}_0$ from the real unknown and possibily complex data distribution $q(\mathbf{x}_0)$
# * we sample a noise level $t$ uniformally between $1$ and $T$ (i.e., a random time step)
# * we sample some noise from a Gaussian distribution and corrupt the input by this noise at level $t$ using the nice property defined above
# * the neural network is trained to predict this noise based on the corruped image $\mathbf{x}_t$, i.e. noise applied on $\mathbf{x}_0$ based on known schedule $\beta_t$

# In reality, all of this is done on batches of data as one uses stochastic gradient descent to optimize neural networks.

# ## The neural network

# The neural network needs to take in a noised image at a particular time step and return the predicted noise. Note that the predicted noise is a tensor that has the same size/resolution as the input image. So technically, the network takes in and outputs tensors of the same shape. What type of neural network can we use for this? 

# What is typically used here is very similar to that of an [Autoencoder](https://en.wikipedia.org/wiki/Autoencoder), which you may remember from typical "intro to deep learning" tutorials. Autoencoders have a so-called "bottleneck" layer in between the encoder and decoder. The encoder first encodes an image into a smaller hidden representation called the "bottleneck", and the decoder then decodes that hidden representation back into an actual image. This forces the network to only keep the most important information in the bottleneck layer.

# In terms of architecture, the DDPM authors went for a **U-Net**, introduced by ([Ronneberger et al., 2015](https://arxiv.org/abs/1505.04597)) (which, at the time, achieved state-of-the-art results for medical image segmentation). This network, like any autoencoder, consists of a bottleneck in the middle that makes sure the network learns only the most important information. Importantly, it introduced residual connections between the encoder and decoder, greatly improving gradient flow (inspired by ResNet in [He et al., 2015](https://arxiv.org/abs/1512.03385)).

# <p align="center">
#     <img src="https://drive.google.com/uc?id=1_Hej_VTgdUWGsxxIuyZACCGjpbCGIUi6" width="400" />
# </p>

# As can be seen, a U-Net model first downsamples the input (i.e. makes the input smaller in terms of spatial resolution), after which upsampling is performed.

# Below, we implement this network, step-by-step.

# ### Network helpers

# First, we define some helper functions and classes which will be used when implementing the neural network. Importantly, we define a `Residual` module, which simply adds the input to the output of a particular function (in other words, adds a residual connection to a particular function).

# We also define aliases for the up- and downsampling operations.
# """

# def exists(x):
#     return x is not None

# def default(val, d):
#     if exists(val):
#         return val
#     return d() if isfunction(d) else d

# class Residual(nn.Module):
#     def __init__(self, fn):
#         super().__init__()
#         self.fn = fn

#     def forward(self, x, *args, **kwargs):
#         return self.fn(x, *args, **kwargs) + x

# def Upsample(dim):
#     return nn.ConvTranspose2d(dim, dim, 4, 2, 1)

# def Downsample(dim):
#     return nn.Conv2d(dim, dim, 4, 2, 1)

# """### Position embeddings

# As the parameters of the neural network are shared across time (noise level), the authors employ sinusoidal position embeddings to encode $t$, inspired by the Transformer ([Vaswani et al., 2017](https://arxiv.org/abs/1706.03762)). This makes the neural network "know" at which particular time step (noise level) it is operating, for every image in a batch.

# The `SinusoidalPositionEmbeddings` module takes a tensor of shape `(batch_size, 1)` as input (i.e. the noise levels of several noisy images in a batch), and turns this into a tensor of shape `(batch_size, dim)`, with `dim` being the dimensionality of the position embeddings. This is then added to each residual block, as we will see further.
# """

# class SinusoidalPositionEmbeddings(nn.Module):
#     def __init__(self, dim):
#         super().__init__()
#         self.dim = dim

#     def forward(self, time):
#         device = time.device
#         half_dim = self.dim // 2
#         embeddings = math.log(10000) / (half_dim - 1)
#         embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
#         embeddings = time[:, None] * embeddings[None, :]
#         embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
#         return embeddings

# """### ResNet/ConvNeXT block

# Next, we define the core building block of the U-Net model. The DDPM authors employed a Wide ResNet block ([Zagoruyko et al., 2016](https://arxiv.org/abs/1605.07146)), but Phil Wang decided to also add support for a ConvNeXT block ([Liu et al., 2022](https://arxiv.org/abs/2201.03545)), as the latter has achieved great success in the image domain. One can choose one or another in the final U-Net architecture.
# """

# class Block(nn.Module):
#     def __init__(self, dim, dim_out, groups = 8):
#         super().__init__()
#         self.proj = nn.Conv2d(dim, dim_out, 3, padding = 1)
#         self.norm = nn.GroupNorm(groups, dim_out)
#         self.act = nn.SiLU()

#     def forward(self, x, scale_shift = None):
#         x = self.proj(x)
#         x = self.norm(x)

#         if exists(scale_shift):
#             scale, shift = scale_shift
#             x = x * (scale + 1) + shift

#         x = self.act(x)
#         return x

# class ResnetBlock(nn.Module):
#     """https://arxiv.org/abs/1512.03385"""
    
#     def __init__(self, dim, dim_out, *, time_emb_dim=None, groups=8):
#         super().__init__()
#         self.mlp = (
#             nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, dim_out))
#             if exists(time_emb_dim)
#             else None
#         )

#         self.block1 = Block(dim, dim_out, groups=groups)
#         self.block2 = Block(dim_out, dim_out, groups=groups)
#         self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

#     def forward(self, x, time_emb=None):
#         h = self.block1(x)

#         if exists(self.mlp) and exists(time_emb):
#             time_emb = self.mlp(time_emb)
#             h = rearrange(time_emb, "b c -> b c 1 1") + h

#         h = self.block2(h)
#         return h + self.res_conv(x)
    
# class ConvNextBlock(nn.Module):
#     """https://arxiv.org/abs/2201.03545"""

#     def __init__(self, dim, dim_out, *, time_emb_dim=None, mult=2, norm=True):
#         super().__init__()
#         self.mlp = (
#             nn.Sequential(nn.GELU(), nn.Linear(time_emb_dim, dim))
#             if exists(time_emb_dim)
#             else None
#         )

#         self.ds_conv = nn.Conv2d(dim, dim, 7, padding=3, groups=dim)

#         self.net = nn.Sequential(
#             nn.GroupNorm(1, dim) if norm else nn.Identity(),
#             nn.Conv2d(dim, dim_out * mult, 3, padding=1),
#             nn.GELU(),
#             nn.GroupNorm(1, dim_out * mult),
#             nn.Conv2d(dim_out * mult, dim_out, 3, padding=1),
#         )

#         self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

#     def forward(self, x, time_emb=None):
#         h = self.ds_conv(x)

#         if exists(self.mlp) and exists(time_emb):
#             assert exists(time_emb), "time embedding must be passed in"
#             condition = self.mlp(time_emb)
#             h = h + rearrange(condition, "b c -> b c 1 1")

#         h = self.net(h)
#         return h + self.res_conv(x)

# """### Attention module

# Next, we define the attention module, which the DDPM authors added in between the convolutional blocks. Attention is the building block of the famous Transformer architecture ([Vaswani et al., 2017](https://arxiv.org/abs/1706.03762)), which has shown great success in various domains of AI, from NLP and vision to [protein folding](https://www.deepmind.com/blog/alphafold-a-solution-to-a-50-year-old-grand-challenge-in-biology). Phil Wang employs 2 variants of attention: one is regular multi-head self-attention (as used in the Transformer), the other one is a [linear attention variant](https://github.com/lucidrains/linear-attention-transformer) ([Shen et al., 2018](https://arxiv.org/abs/1812.01243)), whose time- and memory requirements scale linear in the sequence length, as opposed to quadratic for regular attention.

# For an extensive explanation of the attention mechanism, we refer the reader to Jay Allamar's [wonderful blog post](https://jalammar.github.io/illustrated-transformer/).
# """

# class Attention(nn.Module):
#     def __init__(self, dim, heads=4, dim_head=32):
#         super().__init__()
#         self.scale = dim_head**-0.5
#         self.heads = heads
#         hidden_dim = dim_head * heads
#         self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
#         self.to_out = nn.Conv2d(hidden_dim, dim, 1)

#     def forward(self, x):
#         b, c, h, w = x.shape
#         qkv = self.to_qkv(x).chunk(3, dim=1)
#         q, k, v = map(
#             lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv
#         )
#         q = q * self.scale

#         sim = einsum("b h d i, b h d j -> b h i j", q, k)
#         sim = sim - sim.amax(dim=-1, keepdim=True).detach()
#         attn = sim.softmax(dim=-1)

#         out = einsum("b h i j, b h d j -> b h i d", attn, v)
#         out = rearrange(out, "b h (x y) d -> b (h d) x y", x=h, y=w)
#         return self.to_out(out)

# class LinearAttention(nn.Module):
#     def __init__(self, dim, heads=4, dim_head=32):
#         super().__init__()
#         self.scale = dim_head**-0.5
#         self.heads = heads
#         hidden_dim = dim_head * heads
#         self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)

#         self.to_out = nn.Sequential(nn.Conv2d(hidden_dim, dim, 1), 
#                                     nn.GroupNorm(1, dim))

#     def forward(self, x):
#         b, c, h, w = x.shape
#         qkv = self.to_qkv(x).chunk(3, dim=1)
#         q, k, v = map(
#             lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv
#         )

#         q = q.softmax(dim=-2)
#         k = k.softmax(dim=-1)

#         q = q * self.scale
#         context = torch.einsum("b h d n, b h e n -> b h d e", k, v)

#         out = torch.einsum("b h d e, b h d n -> b h e n", context, q)
#         out = rearrange(out, "b h c (x y) -> b (h c) x y", h=self.heads, x=h, y=w)
#         return self.to_out(out)

# """### Group normalization

# The DDPM authors interleave the convolutional/attention layers of the U-Net with group normalization ([Wu et al., 2018](https://arxiv.org/abs/1803.08494)). Below, we define a `PreNorm` class, which will be used to apply groupnorm before the attention layer, as we'll see further. Note that there's been a [debate](https://tnq177.github.io/data/transformers_without_tears.pdf) about whether to apply normalization before or after attention in Transformers.
# """

# class PreNorm(nn.Module):
#     def __init__(self, dim, fn):
#         super().__init__()
#         self.fn = fn
#         self.norm = nn.GroupNorm(1, dim)

#     def forward(self, x):
#         x = self.norm(x)
#         return self.fn(x)

# """### Conditional U-Net

# Now that we've defined all building blocks (position embeddings, ResNet/ConvNeXT blocks, attention and group normalization), it's time to define the entire neural network. Recall that the job of the network \\(\mathbf{\epsilon}_\theta(\mathbf{x}_t, t)\\) is to take in a batch of noisy images + noise levels, and output the noise added to the input. More formally:

# - the network takes a batch of noisy images of shape `(batch_size, num_channels, height, width)` and a batch of noise levels of shape `(batch_size, 1)` as input, and returns a tensor of shape `(batch_size, num_channels, height, width)`

# The network is built up as follows:
# * first, a convolutional layer is applied on the batch of noisy images, and position embeddings are computed for the noise levels
# * next, a sequence of downsampling stages are applied. Each downsampling stage consists of 2 ResNet/ConvNeXT blocks + groupnorm + attention + residual connection + a downsample operation
# * at the middle of the network, again ResNet or ConvNeXT blocks are applied, interleaved with attention
# * next, a sequence of upsampling stages are applied. Each upsampling stage consists of 2 ResNet/ConvNeXT blocks + groupnorm + attention + residual connection + an upsample operation
# * finally, a ResNet/ConvNeXT block followed by a convolutional layer is applied.

# Ultimately, neural networks stack up layers as if they were lego blocks (but it's important to [understand how they work](http://karpathy.github.io/2019/04/25/recipe/)).

# """

# class Unet(nn.Module):
#     def __init__(
#         self,
#         dim,
#         init_dim=None,
#         out_dim=None,
#         dim_mults=(1, 2, 4, 8),
#         channels=3,
#         with_time_emb=True,
#         resnet_block_groups=8,
#         use_convnext=True,
#         convnext_mult=2,
#     ):
#         super().__init__()

#         # determine dimensions
#         self.channels = channels

#         init_dim = default(init_dim, dim // 3 * 2)
#         self.init_conv = nn.Conv2d(channels, init_dim, 7, padding=3)

#         dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
#         in_out = list(zip(dims[:-1], dims[1:]))
        
#         if use_convnext:
#             block_klass = partial(ConvNextBlock, mult=convnext_mult)
#         else:
#             block_klass = partial(ResnetBlock, groups=resnet_block_groups)

#         # time embeddings
#         if with_time_emb:
#             time_dim = dim * 4
#             self.time_mlp = nn.Sequential(
#                 SinusoidalPositionEmbeddings(dim),
#                 nn.Linear(dim, time_dim),
#                 nn.GELU(),
#                 nn.Linear(time_dim, time_dim),
#             )
#         else:
#             time_dim = None
#             self.time_mlp = None

#         # layers
#         self.downs = nn.ModuleList([])
#         self.ups = nn.ModuleList([])
#         num_resolutions = len(in_out)

#         for ind, (dim_in, dim_out) in enumerate(in_out):
#             is_last = ind >= (num_resolutions - 1)

#             self.downs.append(
#                 nn.ModuleList(
#                     [
#                         block_klass(dim_in, dim_out, time_emb_dim=time_dim),
#                         block_klass(dim_out, dim_out, time_emb_dim=time_dim),
#                         Residual(PreNorm(dim_out, LinearAttention(dim_out))),
#                         Downsample(dim_out) if not is_last else nn.Identity(),
#                     ]
#                 )
#             )

#         mid_dim = dims[-1]
#         self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)
#         self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
#         self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)

#         for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
#             is_last = ind >= (num_resolutions - 1)

#             self.ups.append(
#                 nn.ModuleList(
#                     [
#                         block_klass(dim_out * 2, dim_in, time_emb_dim=time_dim),
#                         block_klass(dim_in, dim_in, time_emb_dim=time_dim),
#                         Residual(PreNorm(dim_in, LinearAttention(dim_in))),
#                         Upsample(dim_in) if not is_last else nn.Identity(),
#                     ]
#                 )
#             )

#         out_dim = default(out_dim, channels)
#         self.final_conv = nn.Sequential(
#             block_klass(dim, dim), nn.Conv2d(dim, out_dim, 1)
#         )

#     def forward(self, x, time):
#         x = self.init_conv(x)

#         t = self.time_mlp(time) if exists(self.time_mlp) else None

#         h = []

#         # downsample
#         for block1, block2, attn, downsample in self.downs:
#             x = block1(x, t)
#             x = block2(x, t)
#             x = attn(x)
#             h.append(x)
#             x = downsample(x)

#         # bottleneck
#         x = self.mid_block1(x, t)
#         x = self.mid_attn(x)
#         x = self.mid_block2(x, t)

#         # upsample
#         for block1, block2, attn, upsample in self.ups:
#             x = torch.cat((x, h.pop()), dim=1)
#             x = block1(x, t)
#             x = block2(x, t)
#             x = attn(x)
#             x = upsample(x)

#         return self.final_conv(x)
    
# def get_optim(dataset: str, model: nn.Module):
#     if dataset == DatasetLoader.MNIST or dataset == DatasetLoader.CIFAR10 or dataset == DatasetLoader.CELEBA:
#         return Adam(model.parameters(), lr=2e-4)
#     elif dataset == DatasetLoader.CELEBA_HQ or dataset == DatasetLoader.LSUN_CHURCH:
#         return Adam(model.parameters(), lr=2e-5)
#     else:
#         raise NotImplementedError(f"Argument dataset: {dataset} isn't defined")

        
# =======================================================================================

from diffusers import UNet2DModel, DDPMScheduler, DDIMScheduler, DPMSolverMultistepScheduler, UniPCMultistepScheduler, PNDMScheduler, DEISMultistepScheduler, HeunDiscreteScheduler, LMSDiscreteScheduler, ScoreSdeVeScheduler, KarrasVeScheduler, DiffusionPipeline, DDPMPipeline, DDIMPipeline, PNDMPipeline, ScoreSdeVePipeline, LDMPipeline, KarrasVePipeline
# from diffusers.scheduling_utils import SchedulerMixin

def batch_sampling(sample_n: int, pipeline, init: torch.Tensor=None, max_batch_n: int=256, rng: torch.Generator=None):
    if init == None:
        if sample_n > max_batch_n:
            replica = sample_n // max_batch_n
            residual = sample_n % max_batch_n
            batch_sizes = [max_batch_n] * (replica) + ([residual] if residual > 0 else [])
        else:
            batch_sizes = [sample_n]
    else:
        init = torch.split(init, max_batch_n)
        batch_sizes = list(map(lambda x: len(x), init))
    sample_imgs_ls = []
    for i, batch_sz in enumerate(batch_sizes):
        pipline_res = pipeline(
                    batch_size=batch_sz, 
                    generator=rng,
                    init=init[i],
                    output_type=None
                )
        sample_imgs_ls.append(pipline_res.images)
    return np.concatenate(sample_imgs_ls)

from typing import Union
import os
from PIL import Image
from tqdm import tqdm

def save_imgs(imgs: np.ndarray, file_dir: Union[str, os.PathLike], file_name: Union[str, os.PathLike]="", start_cnt: int=0) -> None:
        os.makedirs(file_dir, exist_ok=True)
        # Because PIL can only accept 2D matrix for gray-scale images, thus, we need to convert the 3D tensors into 2D ones.
        images = [Image.fromarray(image) for image in np.squeeze((imgs * 255).round().astype("uint8"))]
        for i, img in enumerate(tqdm(images)):
            img.save(os.path.join(file_dir, f"{file_name}{start_cnt + i}.png"))
        del images

def batch_sampling_save(sample_n: int, pipeline, path: Union[str, os.PathLike], init: torch.Tensor=None, max_batch_n: int=256, rng: torch.Generator=None):
    if init == None:
        if sample_n > max_batch_n:
            replica = sample_n // max_batch_n
            residual = sample_n % max_batch_n
            batch_sizes = [max_batch_n] * (replica) + ([residual] if residual > 0 else [])
        else:
            batch_sizes = [sample_n]
    else:
        init = torch.split(init, max_batch_n)
        batch_sizes = list(map(lambda x: len(x), init))
    sample_imgs_ls = []
    cnt = 0
    for i, batch_sz in enumerate(batch_sizes):
        pipline_res = pipeline(
                    batch_size=batch_sz, 
                    generator=rng,
                    init=init[i],
                    output_type=None
                )
        # sample_imgs_ls.append(pipline_res.images)
        save_imgs(imgs=pipline_res.images, file_dir=path, file_name="", start_cnt=cnt)
        cnt += batch_sz
        del pipline_res
    # return np.concatenate(sample_imgs_ls)
    return None
    
class DiffuserModelSched():
    CLIP_SAMPLE_DEFAULT = False
    MODEL_DEFAULT = "DEFAULT"
    DDPM_CIFAR10_DEFAULT = "DDPM-CIFAR10-DEFAULT"
    DDPM_CELEBA_HQ_DEFAULT = "DDPM-CELEBA-HQ-DEFAULT"
    DDPM_CHURCH_DEFAULT = "DDPM-CHURCH-DEFAULT"
    DDPM_BEDROOM_DEFAULT = "DDPM-BEDROOM-DEFAULT"
    LDM_CELEBA_HQ_DEFAULT = "LDM-CELEBA-HQ-DEFAULT"
    
    DDPM_CIFAR10_32 = "DDPM-CIFAR10-32"
    DDPM_CELEBA_HQ_256 = "DDPM-CELEBA-HQ-256"
    DDPM_CHURCH_256 = "DDPM-CHURCH-256"
    DDPM_BEDROOM_256 = "DDPM-BEDROOM-256"
    LDM_CELEBA_HQ_256 = "LDM-CELEBA-HQ-256"

    DDPM_SCHED = "DDPM-SCHED"
    DDIM_SCHED = "DDIM-SCHED"
    DPM_SOLVER_PP_O1_SCHED = "DPM_SOLVER_PP_O1-SCHED"
    DPM_SOLVER_O1_SCHED = "DPM_SOLVER_O1-SCHED"
    DPM_SOLVER_PP_O2_SCHED = "DPM_SOLVER_PP_O2-SCHED"
    DPM_SOLVER_O2_SCHED = "DPM_SOLVER_O2-SCHED"
    DPM_SOLVER_PP_O3_SCHED = "DPM_SOLVER_PP_O3-SCHED"
    DPM_SOLVER_O3_SCHED = "DPM_SOLVER_O3-SCHED"
    UNIPC_SCHED = "UNIPC-SCHED"
    PNDM_SCHED = "PNDM-SCHED"
    DEIS_SCHED = "DEIS-SCHED"
    HEUN_SCHED = "HEUN-SCHED"
    LMSD_SCHED = "LMSD-SCHED"
    LDM_SCHED = "LDM-SCHED"
    SCORE_SDE_VE_SCHED = "SCORE-SDE-VE-SCHED"
    EDM_VE_SCHED = "EDM-VE-SCHED"
    EDM_VE_ODE_SCHED = "EDM-VE-ODE-SCHED"
    EDM_VE_SDE_SCHED = "EDM-VE-SDE-SCHED"

    @staticmethod
    def get_sample_clip(clip_sample: bool, clip_sample_default: bool):
        if clip_sample is not None:
            return clip_sample
        return clip_sample_default
    
    @staticmethod
    def __get_pipeline_generator(unet, scheduler, pipeline):
        def get_pipeline(unet, scheduler):
            return pipeline(unet, scheduler)
        return get_pipeline

    @staticmethod
    def __get_model_sched(ckpt_id: str, clip_sample: bool, noise_sched_type: str=None):
        # Clip option
        clip_sample_used = DiffuserModelSched.get_sample_clip(clip_sample=clip_sample, clip_sample_default=DiffuserModelSched.CLIP_SAMPLE_DEFAULT)
        # Pipeline
        pipline: DDPMPipeline = DDPMPipeline.from_pretrained(ckpt_id)
        
        model: UNet2DModel = pipline.unet
        # noise_sched = pipline.scheduler
        num_train_timesteps: int = 1000
        beta_start: float = 0.0001
        beta_end: float = 0.02
        
        PNDMPipeline_used = partial(PNDMPipeline, clip_sample=clip_sample_used)

        if noise_sched_type == DiffuserModelSched.DDPM_SCHED:
            noise_sched = DDPMScheduler(num_train_timesteps=num_train_timesteps, beta_start=beta_start, beta_end=beta_end, clip_sample=clip_sample_used)
            get_pipeline = DiffuserModelSched.__get_pipeline_generator(unet=model, scheduler=noise_sched, pipeline=DDPMPipeline)
        elif noise_sched_type == DiffuserModelSched.DDIM_SCHED:
            noise_sched = DDIMScheduler(num_train_timesteps=num_train_timesteps, beta_start=beta_start, beta_end=beta_end, clip_sample=clip_sample_used)
            get_pipeline = DiffuserModelSched.__get_pipeline_generator(unet=model, scheduler=noise_sched, pipeline=DDIMPipeline)
        elif noise_sched_type == DiffuserModelSched.DPM_SOLVER_PP_O1_SCHED:
            noise_sched = DPMSolverMultistepScheduler(num_train_timesteps=num_train_timesteps, beta_start=beta_start, beta_end=beta_end, solver_order=1, algorithm_type='dpmsolver++')
            get_pipeline = DiffuserModelSched.__get_pipeline_generator(unet=model, scheduler=noise_sched, pipeline=PNDMPipeline_used)
        elif noise_sched_type == DiffuserModelSched.DPM_SOLVER_O1_SCHED:
            noise_sched = DPMSolverMultistepScheduler(num_train_timesteps=num_train_timesteps, beta_start=beta_start, beta_end=beta_end, solver_order=1, algorithm_type='dpmsolver')
            get_pipeline = DiffuserModelSched.__get_pipeline_generator(unet=model, scheduler=noise_sched, pipeline=PNDMPipeline_used)
        elif noise_sched_type == DiffuserModelSched.DPM_SOLVER_PP_O2_SCHED:
            noise_sched = DPMSolverMultistepScheduler(num_train_timesteps=num_train_timesteps, beta_start=beta_start, beta_end=beta_end, solver_order=2, algorithm_type='dpmsolver++')
            get_pipeline = DiffuserModelSched.__get_pipeline_generator(unet=model, scheduler=noise_sched, pipeline=PNDMPipeline_used)
        elif noise_sched_type == DiffuserModelSched.DPM_SOLVER_O2_SCHED:
            noise_sched = DPMSolverMultistepScheduler(num_train_timesteps=num_train_timesteps, beta_start=beta_start, beta_end=beta_end, solver_order=2, algorithm_type='dpmsolver')
            get_pipeline = DiffuserModelSched.__get_pipeline_generator(unet=model, scheduler=noise_sched, pipeline=PNDMPipeline_used)
        elif noise_sched_type == DiffuserModelSched.DPM_SOLVER_PP_O3_SCHED:
            noise_sched = DPMSolverMultistepScheduler(num_train_timesteps=num_train_timesteps, beta_start=beta_start, beta_end=beta_end, solver_order=3, algorithm_type='dpmsolver++')
            get_pipeline = DiffuserModelSched.__get_pipeline_generator(unet=model, scheduler=noise_sched, pipeline=PNDMPipeline_used)
        elif noise_sched_type == DiffuserModelSched.DPM_SOLVER_O3_SCHED:
            noise_sched = DPMSolverMultistepScheduler(num_train_timesteps=num_train_timesteps, beta_start=beta_start, beta_end=beta_end, solver_order=3, algorithm_type='dpmsolver')
            get_pipeline = DiffuserModelSched.__get_pipeline_generator(unet=model, scheduler=noise_sched, pipeline=PNDMPipeline_used)
        elif noise_sched_type == DiffuserModelSched.UNIPC_SCHED:
            noise_sched = UniPCMultistepScheduler(num_train_timesteps=num_train_timesteps, beta_start=beta_start, beta_end=beta_end)
            get_pipeline = DiffuserModelSched.__get_pipeline_generator(unet=model, scheduler=noise_sched, pipeline=PNDMPipeline_used)
        elif noise_sched_type == DiffuserModelSched.PNDM_SCHED:
            noise_sched = PNDMScheduler(num_train_timesteps=num_train_timesteps, beta_start=beta_start, beta_end=beta_end)
            get_pipeline = DiffuserModelSched.__get_pipeline_generator(unet=model, scheduler=noise_sched, pipeline=PNDMPipeline_used)
        elif noise_sched_type == DiffuserModelSched.DEIS_SCHED:
            noise_sched = DEISMultistepScheduler(num_train_timesteps=num_train_timesteps, beta_start=beta_start, beta_end=beta_end)
            get_pipeline = DiffuserModelSched.__get_pipeline_generator(unet=model, scheduler=noise_sched, pipeline=PNDMPipeline_used)
        elif noise_sched_type == DiffuserModelSched.HEUN_SCHED:
            noise_sched = HeunDiscreteScheduler(num_train_timesteps=num_train_timesteps, beta_start=beta_start, beta_end=beta_end)
            get_pipeline = DiffuserModelSched.__get_pipeline_generator(unet=model, scheduler=noise_sched, pipeline=PNDMPipeline_used)
        elif noise_sched_type == DiffuserModelSched.LMSD_SCHED:
            noise_sched = LMSDiscreteScheduler(num_train_timesteps=num_train_timesteps, beta_start=beta_start, beta_end=beta_end)
            get_pipeline = DiffuserModelSched.__get_pipeline_generator(unet=model, scheduler=noise_sched, pipeline=PNDMPipeline_used)
        elif noise_sched_type == None:
            noise_sched = pipline.scheduler
            get_pipeline = DiffuserModelSched.__get_pipeline_generator(unet=model, scheduler=noise_sched, pipeline=DDPMPipeline)
            # noise_sched = DDPMScheduler.from_pretrained(ckpt_id, prediction_type='epsilon')
            # noise_sched =DDPMScheduler(num_train_timesteps=1000, beta_start=0.0001, beta_end=0.02)
        else:
            raise NotImplementedError()
        
        if clip_sample_used != None:
            noise_sched.config.clip_sample = clip_sample_used
            print(f"noise_sched.config.clip_sample = {noise_sched.config.clip_sample}")
            
        return model, noise_sched, get_pipeline
            
    @staticmethod
    def get_model_sched(image_size: int, channels: int, model_type: str=MODEL_DEFAULT, noise_sched_type: str=None, clip_sample: bool=None, **kwargs):
        @torch.no_grad()
        def weight_reset(m: nn.Module):
            # - check if the current module has reset_parameters & if it's callabed called it on m
            reset_parameters = getattr(m, "reset_parameters", None)
            if callable(reset_parameters):
                m.reset_parameters()
            
        if model_type == DiffuserModelSched.MODEL_DEFAULT:
            clip_sample_used = DiffuserModelSched.get_sample_clip(clip_sample=clip_sample, clip_sample_default=False)
            noise_sched = DDPMScheduler(num_train_timesteps=1000, tensor_format="pt", clip_sample=clip_sample_used)
            model = UNet2DModel(
                sample_size=image_size,  # the target image resolution
                in_channels=channels,  # the number of input channels, 3 for RGB images
                out_channels=channels,  # the number of output channels
                layers_per_block=2,  # how many ResNet layers to use per UNet block
                block_out_channels=(128, 128, 256, 256, 512, 512),  # the number of output channes for each UNet block
                down_block_types=( 
                    "DownBlock2D",  # a regular ResNet downsampling block
                    "DownBlock2D", 
                    "DownBlock2D", 
                    "DownBlock2D", 
                    "AttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
                    "DownBlock2D",
                ), 
                up_block_types=(
                    "UpBlock2D",  # a regular ResNet upsampling block
                    "AttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
                    "UpBlock2D", 
                    "UpBlock2D", 
                    "UpBlock2D", 
                    "UpBlock2D"  
                ),
            )
            get_pipeline = DiffuserModelSched.__get_pipeline_generator(unet=model, scheduler=noise_sched, pipeline=DDPMPipeline)
        elif model_type == DiffuserModelSched.DDPM_CIFAR10_DEFAULT:
            model, noise_sched, get_pipeline = DiffuserModelSched.get_pretrained(ckpt=DiffuserModelSched.DDPM_CIFAR10_32, noise_sched_type=noise_sched_type, clip_sample=clip_sample)
            model = model.apply(weight_reset)
        elif model_type == DiffuserModelSched.DDPM_CELEBA_HQ_DEFAULT:
            model, noise_sched, get_pipeline = DiffuserModelSched.get_pretrained(ckpt=DiffuserModelSched.DDPM_CELEBA_HQ_256, noise_sched_type=noise_sched_type, clip_sample=clip_sample)
            model = model.apply(weight_reset)
        elif model_type == DiffuserModelSched.DDPM_CHURCH_DEFAULT:
            model, noise_sched, get_pipeline = DiffuserModelSched.get_pretrained(ckpt=DiffuserModelSched.DDPM_CHURCH_256, noise_sched_type=noise_sched_type, clip_sample=clip_sample)
            model = model.apply(weight_reset)
        elif model_type == DiffuserModelSched.DDPM_BEDROOM_DEFAULT:
            model, noise_sched, get_pipeline = DiffuserModelSched.get_pretrained(ckpt=DiffuserModelSched.DDPM_BEDROOM_256, noise_sched_type=noise_sched_type, clip_sample=clip_sample)
            model = model.apply(weight_reset)
        elif model_type == DiffuserModelSched.LDM_CELEBA_HQ_DEFAULT:
            model, noise_sched, get_pipeline = DiffuserModelSched.get_pretrained(ckpt=DiffuserModelSched.LDM_CELEBA_HQ_256, noise_sched_type=noise_sched_type, clip_sample=clip_sample)
            model = model.apply(weight_reset)
        else:
            raise NotImplementedError()
        return model, noise_sched, get_pipeline
    
    @staticmethod
    def get_pretrained(ckpt: str, clip_sample: bool=None, noise_sched_type: str=None):        
        if ckpt == DiffuserModelSched.DDPM_CIFAR10_32:
            ckpt: str = "google/ddpm-cifar10-32"
        elif ckpt == DiffuserModelSched.DDPM_CELEBA_HQ_256:
            ckpt: str = "google/ddpm-ema-celebahq-256"
        elif ckpt == DiffuserModelSched.DDPM_CHURCH_256:
            ckpt: str = "google/ddpm-ema-church-256"
        elif ckpt == DiffuserModelSched.DDPM_BEDROOM_256:
            ckpt: str = "google/ddpm-ema-bedroom-256"
        elif ckpt == DiffuserModelSched.LDM_CELEBA_HQ_256:
            ckpt: str = "CompVis/ldm-celebahq-256"
        # else:
        #     raise NotImplementedError(f"Argument ckpt: {ckpt} isn't defined")
        # # Clip option
        # clip_sample_used = DiffuserModelSched.get_sample_clip(clip_sample=clip_sample, clip_sample_default=False)
        # # Pipeline
        # pipline: DDPMPipeline = DDPMPipeline.from_pretrained(pretrained_id)
        
        # model: UNet2DModel = pipline.unet
        # noise_sched = pipline.scheduler
        # if clip_sample_used != None:
        #     noise_sched.config.clip_sample = clip_sample_used
            
        # return model, noise_sched
        return DiffuserModelSched.__get_model_sched(ckpt_id=ckpt, clip_sample=clip_sample, noise_sched_type=noise_sched_type)
    
    @staticmethod
    def get_trained(ckpt: str, clip_sample: bool=None, noise_sched_type: str=None):        
        return DiffuserModelSched.__get_model_sched(ckpt_id=ckpt, clip_sample=clip_sample, noise_sched_type=noise_sched_type)
        