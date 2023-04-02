# %%
from functools import partial
from os import terminal_size
from sched import scheduler

import torch
from torch import nn
import torch.nn.functional as F

from dataset import Backdoor, DEFAULT_VMIN, DEFAULT_VMAX

# """## Defining the forward diffusion process

# The forward diffusion process gradually adds noise to an image from the real distribution, in a number of time steps $T$. This happens according to a **variance schedule**. The original DDPM authors employed a linear schedule:

# > We set the forward process variances to constants
# increasing linearly from $\beta_1 = 10^{−4}$
# to $\beta_T = 0.02$.

# However, it was shown in ([Nichol et al., 2021](https://arxiv.org/abs/2102.09672)) that better results can be achieved when employing a cosine schedule. 

# Below, we define various schedules for the $T$ timesteps, as well as corresponding variables which we'll need, such as cumulative variances.
# """

# def cosine_beta_schedule(timesteps, s=0.008):
#     """
#     cosine schedule as proposed in https://arxiv.org/abs/2102.09672
#     """
#     steps = timesteps + 1
#     x = torch.linspace(0, timesteps, steps)
#     alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
#     alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
#     betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
#     return torch.clip(betas, 0.0001, 0.9999)

# def linear_beta_schedule(timesteps):
#     beta_start = 0.0001
#     beta_end = 0.02
#     return torch.linspace(beta_start, beta_end, timesteps)

# def quadratic_beta_schedule(timesteps):
#     beta_start = 0.0001
#     beta_end = 0.02
#     return torch.linspace(beta_start**0.5, beta_end**0.5, timesteps) ** 2

# def sigmoid_beta_schedule(timesteps):
#     beta_start = 0.0001
#     beta_end = 0.02
#     betas = torch.linspace(-6, 6, timesteps)
#     return torch.sigmoid(betas) * (beta_end - beta_start) + beta_start

# def extract(a, t, x_shape):
#     batch_size = t.shape[0]
#     out = a.gather(-1, t.cpu())
#     return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

# class NoiseScheduler():
#     SCHED_COSINE = "SC_COS"
#     SCHED_LINEAR = "SC_LIN"
#     SCHED_QUADRATIC = "SC_QUAD"
#     SCHED_SIGMOID = "SC_SIGM"
#     def __init__(self, timesteps: int, scheduler: str, s: float=0.008):
#         self.__timesteps = int(timesteps)
#         self.__s = float(s)
#         self.__scheduler = scheduler
        
#         # define beta schedule
#         if self.__scheduler == self.SCHED_COSINE:
#             self.__betas = NoiseScheduler.cosine_beta_schedule(timesteps=self.__timesteps, s=self.__s)
#         elif self.__scheduler == self.SCHED_LINEAR:
#             self.__betas = NoiseScheduler.linear_beta_schedule(timesteps=self.__timesteps)
#         elif self.__scheduler == self.SCHED_QUADRATIC:
#             self.__betas = NoiseScheduler.quadratic_beta_schedule(timesteps=self.__timesteps)
#         elif self.__scheduler == self.SCHED_SIGMOID:
#             self.__betas = NoiseScheduler.sigmoid_beta_schedule(timesteps=self.__timesteps)
#         else:
#             raise ImportError(f"Undefined scheduler: {self.__scheduler}")
            
#         # define alphas 
#         self.__alphas = 1. - self.betas
#         self.__alphas_cumprod = torch.cumprod(self.alphas, axis=0)
#         self.__alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
#         self.__sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)

#         # Calculations for backdoor
#         self.__sqrt_alphas = torch.sqrt(self.alphas)
#         self.__one_minus_sqrt_alphas = 1 - self.sqrt_alphas
#         self.__one_minus_alphas = 1 - self.alphas

#         # calculations for diffusion q(x_t | x_{t-1}) and others
#         self.__sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
#         self.__sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
#         self.__R_coef = self.one_minus_sqrt_alphas * self.sqrt_one_minus_alphas_cumprod / self.one_minus_alphas

#         # calculations for posterior q(x_{t-1} | x_t, x_0)
#         self.__posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        
#     @staticmethod
#     def cosine_beta_schedule(timesteps, s=0.008):
#         """
#         cosine schedule as proposed in https://arxiv.org/abs/2102.09672
#         """
#         steps = timesteps + 1
#         x = torch.linspace(0, timesteps, steps)
#         alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
#         alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
#         betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
#         return torch.clip(betas, 0.0001, 0.9999)
#     @staticmethod
#     def linear_beta_schedule(timesteps):
#         beta_start = 0.0001
#         beta_end = 0.02
#         return torch.linspace(beta_start, beta_end, timesteps)
#     @staticmethod
#     def quadratic_beta_schedule(timesteps):
#         beta_start = 0.0001
#         beta_end = 0.02
#         return torch.linspace(beta_start**0.5, beta_end**0.5, timesteps) ** 2
#     @staticmethod
#     def sigmoid_beta_schedule(timesteps):
#         beta_start = 0.0001
#         beta_end = 0.02
#         betas = torch.linspace(-6, 6, timesteps)
#         return torch.sigmoid(betas) * (beta_end - beta_start) + beta_start
#     @property
#     def betas(self):
#         return self.__betas
#     @property
#     def alphas(self):
#         return self.__alphas
#     @property
#     def alphas_cumprod(self):
#         return self.__alphas_cumprod
#     @property
#     def alphas_cumprod_prev(self):
#         return self.__alphas_cumprod_prev
#     @property
#     def sqrt_recip_alphas(self):
#         return self.__sqrt_recip_alphas
#     @property
#     def sqrt_alphas(self):
#         return self.__sqrt_alphas
#     @property
#     def one_minus_sqrt_alphas(self):
#         return self.__one_minus_sqrt_alphas
#     @property
#     def one_minus_alphas(self):
#         return self.__one_minus_alphas
#     @property
#     def sqrt_alphas_cumprod(self):
#         return self.__sqrt_alphas_cumprod
#     @property
#     def sqrt_one_minus_alphas_cumprod(self):
#         return self.__sqrt_one_minus_alphas_cumprod
#     @property
#     def R_coef(self):
#         return self.__R_coef
#     @property
#     def posterior_variance(self):
#         return self.__posterior_variance

# """<img src="https://drive.google.com/uc?id=1QifsBnYiijwTqru6gur9C0qKkFYrm-lN" width="800" />
    
# This means that we can now define the loss function given the model as follows:
# """

# # forward diffusion
# def q_sample_clean(noise_sched, x_start, t, noise=None):
#     if noise is None:
#         noise = torch.randn_like(x_start)

#     sqrt_alphas_cumprod_t = extract(noise_sched.sqrt_alphas_cumprod, t, x_start.shape)
#     sqrt_one_minus_alphas_cumprod_t = extract(
#         noise_sched.sqrt_one_minus_alphas_cumprod, t, x_start.shape
#     )

#     return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise, noise

# def q_sample_backdoor(noise_sched, x_start, R, t, noise=None):
#     if noise is None:
#         noise = torch.randn_like(x_start)

#     sqrt_alphas_cumprod_t = extract(noise_sched.sqrt_alphas_cumprod, t, x_start.shape)
#     sqrt_one_minus_alphas_cumprod_t = extract(
#         noise_sched.sqrt_one_minus_alphas_cumprod, t, x_start.shape
#     )
#     R_coef_t = extract(noise_sched.R_coef, t, x_start.shape)

#     return sqrt_alphas_cumprod_t * x_start + (1 - sqrt_alphas_cumprod_t) * R + sqrt_one_minus_alphas_cumprod_t * noise, R_coef_t * R + noise 

# """
# <img src="https://drive.google.com/uc?id=1QifsBnYiijwTqru6gur9C0qKkFYrm-lN" width="800" />    
# This means that we can now define the loss function given the model as follows:
# """

# def p_losses_clean(noise_sched, denoise_model, x_start, t, noise=None, loss_type="l2"):
#     if len(x_start) == 0: 
#         return 0
#     if noise is None:
#         noise = torch.randn_like(x_start)

#     x_noisy, target = q_sample_clean(noise_sched=noise_sched, x_start=x_start, t=t, noise=noise)
#     predicted_noise = denoise_model(x_noisy, t)

#     if loss_type == 'l1':
#         loss = F.l1_loss(target, predicted_noise)
#     elif loss_type == 'l2':
#         loss = F.mse_loss(target, predicted_noise)
#     elif loss_type == "huber":
#         loss = F.smooth_l1_loss(target, predicted_noise)
#     else:
#         raise NotImplementedError()

#     return loss

# def p_losses_backdoor(noise_sched, denoise_model, x_start, R, t, noise=None, loss_type="l2"):
#     if len(x_start) == 0: 
#         return 0
#     if noise is None:
#         noise = torch.randn_like(x_start)

#     x_noisy, target = q_sample_backdoor(noise_sched=noise_sched, x_start=x_start, R=R, t=t, noise=noise)
#     predicted_noise = denoise_model(x_noisy, t)

#     if loss_type == 'l1':
#         loss = F.l1_loss(target, predicted_noise)
#     elif loss_type == 'l2':
#         loss = F.mse_loss(target, predicted_noise)
#     elif loss_type == "huber":
#         loss = F.smooth_l1_loss(target, predicted_noise)
#     else:
#         raise NotImplementedError()

#     return loss

# def p_losses(noise_sched, denoise_model, x_start, R, is_clean, t, noise=None, loss_type="l2"):
#     is_not_clean = torch.where(is_clean, False, True)
#     if noise != None:
#         noise_clean = noise[is_clean]
#         noise_backdoor = noise[is_not_clean]
#     else:
#         noise_clean = noise_backdoor = noise
#     loss_clean = p_losses_clean(noise_sched=noise_sched, denoise_model=denoise_model, x_start=x_start[is_clean], t=t[is_clean], noise=noise_clean, loss_type=loss_type)
#     loss_backdoor = p_losses_backdoor(noise_sched=noise_sched, denoise_model=denoise_model, x_start=x_start[is_not_clean], R=R[is_not_clean], t=t[is_not_clean], noise=noise_backdoor, loss_type=loss_type)

#     return (loss_clean + loss_backdoor) / 2


# # ==================================================
# class LossSampler():
#     def __init__(self, noise_sched: NoiseScheduler):
#         self.__noise_sched = noise_sched

#     def get_fn(self):
#         return partial(p_losses_backdoor, self.__noise_sched), partial(q_sample_backdoor, self.__noise_sched)
    
def q_sample_diffuser(noise_sched, x_start: torch.Tensor, R: torch.Tensor, timesteps: torch.Tensor, noise: torch.Tensor=None) -> torch.Tensor:
    if noise is None:
        noise = torch.randn_like(x_start)
        
    def unqueeze_n(x):
        return x.reshape(len(x_start), *([1] * len(x_start.shape[1:])))

    alphas_cumprod = noise_sched.alphas_cumprod.to(device=x_start.device, dtype=x_start.dtype)
    alphas = noise_sched.alphas.to(device=x_start.device, dtype=x_start.dtype)
    timesteps = timesteps.to(x_start.device)

    sqrt_alphas_cumprod_t = alphas_cumprod[timesteps] ** 0.5
    sqrt_one_minus_alphas_cumprod_t = (1 - alphas_cumprod[timesteps]) ** 0.5
    R_coef_t = (1 - alphas[timesteps] ** 0.5) * sqrt_one_minus_alphas_cumprod_t / (1 - alphas[timesteps])
    
    sqrt_alphas_cumprod_t = unqueeze_n(sqrt_alphas_cumprod_t)
    R_coef_t = unqueeze_n(R_coef_t)
    
    noisy_images = noise_sched.add_noise(x_start, noise, timesteps)

    # return sqrt_alphas_cumprod_t * x_start + (1 - sqrt_alphas_cumprod_t) * R + sqrt_one_minus_alphas_cumprod_t * noise, R_coef_t * R + noise 
    # print(f"x_start shape: {x_start.shape}")
    # print(f"R shape: {R.shape}")
    # print(f"timesteps shape: {timesteps.shape}")
    # print(f"noise shape: {noise.shape}")
    # print(f"noisy_images shape: {noisy_images.shape}")
    # print(f"sqrt_alphas_cumprod_t shape: {sqrt_alphas_cumprod_t.shape}")
    # print(f"R_coef_t shape: {R_coef_t.shape}")
    return noisy_images + (1 - sqrt_alphas_cumprod_t) * R, R_coef_t * R + noise 

def p_losses_diffuser(noise_sched, model: nn.Module, x_start: torch.Tensor, R: torch.Tensor, timesteps: torch.Tensor, noise: torch.Tensor=None, loss_type: str="l2") -> torch.Tensor:
    if len(x_start) == 0: 
        return 0
    if noise is None:
        noise = torch.randn_like(x_start)

    x_noisy, target = q_sample_diffuser(noise_sched=noise_sched, x_start=x_start, R=R, timesteps=timesteps, noise=noise)
    # if clip:
    #     x_noisy = torch.clamp(x_noisy, min=DEFAULT_VMIN, max=DEFAULT_VMAX)
    predicted_noise = model(x_noisy.contiguous(), timesteps.contiguous(), return_dict=False)[0]

    if loss_type == 'l1':
        loss = F.l1_loss(target, predicted_noise)
    elif loss_type == 'l2':
        loss = F.mse_loss(target, predicted_noise)
    elif loss_type == "huber":
        loss = F.smooth_l1_loss(target, predicted_noise)
    else:
        raise NotImplementedError()

    return loss
# %%
if __name__ == '__main__':
    # You can use the following code to visualize the forward process
    import os
    
    from diffusers import DDPMScheduler
    
    from dataset import DatasetLoader
    from model import DiffuserModelSched
    
    time_step = 95
    num_train_timesteps = 100
    # time_step = 140
    # num_train_timesteps = 150
    ds_root = os.path.join('datasets')
    dsl = DatasetLoader(root=ds_root, name=DatasetLoader.CELEBA_HQ).set_poison(trigger_type=Backdoor.TRIGGER_GLASSES, target_type=Backdoor.TARGET_CAT, clean_rate=1, poison_rate=0.2).prepare_dataset()
    print(f"Full Dataset Len: {len(dsl)}")
    image_size = dsl.image_size
    channels = dsl.channel
    ds = dsl.get_dataset()
    # CIFAR10
    # sample = ds[50000]
    # MNIST
    # sample = ds[60000]
    # CelebA-HQ
    # sample = ds[10000]
    sample = ds[24000]
    
    target = torch.unsqueeze(sample[DatasetLoader.TARGET], dim=0)
    source = torch.unsqueeze(sample[DatasetLoader.PIXEL_VALUES], dim=0)
    bs = len(source)
    model, noise_sched = DiffuserModelSched.get_model_sched(image_size=image_size, channels=channels, model_type=DiffuserModelSched.MODEL_DEFAULT)
    
    print(f"bs: {bs}")
    
    # Sample a random timestep for each image
    # mx_timestep = noise_sched.num_train_timesteps
    # timesteps = torch.randint(0, mx_timestep, (bs,), device=source.device).long()
    timesteps = torch.tensor([time_step] * bs, device=source.device).long()
    
    
    print(f"target Shape: {target.shape}")
    dsl.show_sample(img=target[0])
    print(f"source Shape: {source.shape}")
    dsl.show_sample(img=source[0])
    
    noise = torch.randn_like(target)
    noise_sched = DDPMScheduler(num_train_timesteps=num_train_timesteps)
    noisy_images = noise_sched.add_noise(source, noise, timesteps)
    
    noisy_x, target_x = q_sample_diffuser(noise_sched, x_start=target, R=source, timesteps=timesteps, noise=noise)
    
    print(f"target_x Shape: {target_x.shape}")
    dsl.show_sample(img=target_x[0], vmin=torch.min(target_x), vmax=torch.max(target_x))
    print(f"noisy_x Shape: {noisy_x.shape}")
    dsl.show_sample(img=noisy_x[0], vmin=torch.min(noisy_x), vmax=torch.max(noisy_x))
    print(f"source Shape: {source.shape}")
    dsl.show_sample(img=source[0], vmin=torch.min(source), vmax=torch.max(source))
    diff = (noisy_x - source)
    print(f"noisy_x - source Shape: {diff.shape}")
    dsl.show_sample(img=diff[0], vmin=torch.min(diff), vmax=torch.max(diff))
    diff = (target_x - noise)
    print(f"target_x - noise Shape: {diff.shape}")
    dsl.show_sample(img=diff[0], vmin=torch.min(diff), vmax=torch.max(diff))
    
    print(f"noisy_images Shape: {noisy_images.shape}")
    dsl.show_sample(img=noisy_images[0], vmin=torch.min(noisy_images), vmax=torch.max(noisy_images))
    diff_x = noisy_x - noisy_images
    print(f"noisy_x - noisy_images Shape: {diff_x.shape}")
    dsl.show_sample(img=diff_x[0], vmin=torch.min(diff_x), vmax=torch.max(diff_x))
    
    diff_x = noisy_x - target_x
    print(f"noisy_x - target_x Shape: {diff_x.shape}")
    dsl.show_sample(img=diff_x[0], vmin=torch.min(diff_x), vmax=torch.max(diff_x))# %%

# # %%
