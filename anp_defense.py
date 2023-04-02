from anp_config import get_config, Config    
config = get_config()

import os
import traceback
from typing import Dict, Union

import torch
from torch import nn, Tensor
from torch.nn import functional as F
from torchinfo import summary
from torchmetrics import StructuralSimilarityIndexMeasure
from accelerate import Accelerator
from tqdm import tqdm

from diffusers import DDPMPipeline

from model import DiffuserModelSched, batch_sampling
from dataset import DatasetLoader, Backdoor, ImagePathDataset
from loss import p_losses_diffuser, q_sample_diffuser

from anp_util import Config, get_model_optim_sched, get_accelerator, sampling, get_data_loader, init_tracker, save_imgs, update_score_file, log_score
from util import Log

def inspect_model(model: nn.Module):
    layers = list(model.children())[:]
    for i, layer in enumerate(layers):
        print(f"[{i}]: {layer}")

def init_training(config: Config, data_loader, dataset_loader: DatasetLoader):
    
    accelerator = get_accelerator(config=config)
    init_tracker(config=config, accelerator=accelerator)
    
    model, perturb_model, optim, noise_sched, lr_sched = get_model_optim_sched(config=config, dataset_loader=dataset_loader)
    
    if lr_sched != None:
        perturb_model, optim, data_loader, lr_sched = accelerator.prepare(
            perturb_model, optim, data_loader, lr_sched
        )
    else:
        perturb_model, optim, data_loader = accelerator.prepare(
            perturb_model, optim, data_loader
        )
    return accelerator, model, perturb_model, optim, data_loader, noise_sched, lr_sched

def backdoor_mse_fn(noise_sched, model: nn.Module, x_start: torch.Tensor, backdoor_x_start: torch.Tensor, R: torch.Tensor, backdoor_R: torch.Tensor, timesteps: torch.Tensor, noise: torch.Tensor=None, loss_type: str="l2") -> torch.Tensor:
    if len(x_start) == 0: 
        return 0
    if noise is None:
        noise = torch.randn_like(x_start)

    x_noisy, target = q_sample_diffuser(noise_sched=noise_sched, x_start=x_start, R=R, timesteps=timesteps, noise=noise)
    backdoor_x_noisy, backdoor_target = q_sample_diffuser(noise_sched=noise_sched, x_start=backdoor_x_start, R=backdoor_R, timesteps=timesteps, noise=noise)
    predicted_noise = model(x_noisy.contiguous(), timesteps.contiguous(), return_dict=False)[0]

    if loss_type == 'l1':
        loss = F.l1_loss(backdoor_target, predicted_noise)
    elif loss_type == 'l2':
        loss = F.mse_loss(backdoor_target, predicted_noise)
    elif loss_type == "huber":
        loss = F.smooth_l1_loss(backdoor_target, predicted_noise)
    else:
        raise NotImplementedError()

    return loss

def clip_weight(model, budget: float=None):
    if budget == None or budget < 0:
        return 
    lower, upper = -budget, budget
    params = [param for name, param in model.named_parameters() if 'bn' in name]
    with torch.no_grad():
        for param in params:
            param.clamp_(lower, upper)
            
def measure(config: Config, accelerator: Accelerator, pipeline, dataset_loader: DatasetLoader, epoch: int=None):
    # Random Number Generator
    rng = torch.Generator()
    rng.manual_seed(config.seed)
    
    # Get epoch & step
    epoch = epoch + 1 if epoch != None else config.epoch
    step = dataset_loader.num_batch * epoch
    
    # Save under folder
    path = os.path.join(config.measure_dir, f"ep{epoch}")
    
    # Generate Samples and Save
    noise = torch.randn(
                (config.measure_sample_n, pipeline.unet.in_channels, pipeline.unet.sample_size, pipeline.unet.sample_size),
                generator=torch.manual_seed(config.seed),
            )
    clean_sample_imgs = batch_sampling(sample_n=config.measure_sample_n, pipeline=pipeline, init=noise, rng=rng)
    save_imgs(imgs=clean_sample_imgs, file_dir=path, file_name="")
    
    # Load Generated Samples
    device = torch.device(config.device_ids[0])
    gen_backdoor_target = ImagePathDataset(path=path)[:].to(device)
    
    reps = ([len(gen_backdoor_target)] + ([1] * (len(dataset_loader.target.shape))))
    backdoor_target = torch.squeeze((dataset_loader.target.repeat(*reps) / 2 + 0.5).clamp(0, 1)).to(device)
    
    # print(f"gen_backdoor_target: {gen_backdoor_target.shape}, vmax: {torch.max(gen_backdoor_target)}, vmin: {torch.min(backdoor_target)} | backdoor_target: {backdoor_target.shape}, vmax: {torch.max(backdoor_target)}, vmin: {torch.min(backdoor_target)}")
    mse_sc = float(nn.MSELoss(reduction='mean')(gen_backdoor_target, backdoor_target))
    ssim_sc = float(StructuralSimilarityIndexMeasure(data_range=1.0).to(device)(gen_backdoor_target, backdoor_target))
    print(f"[{epoch}] MSE: {mse_sc}, SSIM: {ssim_sc}")
    
    sc = update_score_file(config=config, mse_sc=mse_sc, ssim_sc=ssim_sc, epoch=epoch)
    log_score(config=config, accelerator=accelerator, scores=sc, step=step)
    
    return mse_sc, ssim_sc

def train_loop(config: Config, accelerator: Accelerator, model: nn.Module, noise_sched, optim: torch.optim, loader, dataset_loader: DatasetLoader, lr_sched=None, start_epoch: int=0, start_step: int=0):
    # try:
    cur_step = start_step
    epoch = start_epoch
    
    # Test evaluate
    # pipeline = DDPMPipeline(unet=accelerator.unwrap_model(model), scheduler=noise_sched)        
    # sampling(config, 0, pipeline)

    # Now you train the model
    for epoch in range(int(start_epoch), int(config.epoch)):
        progress_bar = tqdm(total=len(loader), disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")

        for step, batch in enumerate(loader):
            # memlog.append()
            # trigger_images = batch['pixel_values'].to(model.device_ids[0])
            # target_images = batch["target"].to(model.device_ids[0])
            clean_images = batch['image']
            trigger_images = batch['pixel_values']
            target_images = batch["target"]
            # Sample noise to add to the images
            noise = torch.randn(trigger_images.shape).to(trigger_images.device)
            bs = trigger_images.shape[0]

            # Sample a random timestep for each image
            timesteps = torch.randint(0, noise_sched.num_train_timesteps, (bs,), device=trigger_images.device).long()

            # Add noise to the clean images according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            
            with accelerator.accumulate(model):
                # Predict the noise residual
                loss = -p_losses_diffuser(noise_sched, model=model, x_start=clean_images, R=torch.full_like(trigger_images, 0), timesteps=timesteps, noise=noise, loss_type="l2")
                accelerator.backward(loss)
                
                # clip_grad_norm_: https://huggingface.co/docs/accelerate/v0.13.2/en/package_reference/accelerator#accelerate.Accelerator.clip_grad_norm_
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optim.step()
                if lr_sched != None:
                    lr_sched.step()
                optim.zero_grad()
            clip_weight(model=model, budget=config.perturb_budget)
            
            progress_bar.update(1)
            with torch.no_grad():
                # backdoor_images = backdoor_ds[:len(trigger_images)]["pixel_values"].to(model.device_ids[0])
                # backdoor_targets = backdoor_ds[:len(trigger_images)]["target"].to(model.device_ids[0])
                backdoor_mse = backdoor_mse_fn(noise_sched, model=model, x_start=clean_images, backdoor_x_start=target_images, R=torch.full_like(trigger_images, 0), backdoor_R=trigger_images, timesteps=timesteps, noise=noise, loss_type="l2")
                
            logs = {"loss": loss.detach().item(), "backdoor_mse": backdoor_mse.detach().item(), "clean_mse": -loss.detach().item(), "epoch": epoch, "step": cur_step}
            if lr_sched != None:
                logs["lr"] = lr_sched.get_last_lr()[0]
            else:
                logs["lr"] = optim.param_groups[0]['lr']
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=cur_step)
            cur_step += 1

        # After each epoch you optionally sample some demo images with evaluate() and save the model
        if accelerator.is_main_process:
            pipeline = DDPMPipeline(unet=accelerator.unwrap_model(model), scheduler=noise_sched)

            if (epoch + 1) % config.save_image_epochs == 0:
                sampling(config, epoch, pipeline)
                measure(config=config, accelerator=accelerator, pipeline=pipeline, dataset_loader=dataset_loader, epoch=epoch)
                
    print(Log.info("Save model and sample images"))
    pipeline = DDPMPipeline(unet=accelerator.unwrap_model(model), scheduler=noise_sched)        
    if accelerator.is_main_process:
        # checkpoint(config=config, accelerator=accelerator, pipeline=pipeline, cur_epoch=epoch, cur_step=cur_step, repo=repo, commit_msg=f"Epoch {epoch}")
        sampling(config, 'final', pipeline)
        measure(config=config, accelerator=accelerator, pipeline=pipeline, dataset_loader=dataset_loader, epoch=None)
    return pipeline

if __name__ == "__main__":
    print(f"PyTorch detected number of availabel devices: {torch.cuda.device_count()}")
    dsl, data_loader, backdoor_dsl, backdoor_loader = get_data_loader(config=config)
    accelerator, model, perturb_model, optim, data_loader, noise_sched, lr_sched = init_training(config=config, data_loader=data_loader, dataset_loader=dsl)
        
    train_loop(config=config, model=perturb_model, accelerator=accelerator, optim=optim, noise_sched=noise_sched, loader=data_loader, dataset_loader=dsl, lr_sched=lr_sched)
    
"""
# References
1. [How can l load my best model as a feature extractor/evaluator?](https://discuss.pytorch.org/t/how-can-l-load-my-best-model-as-a-feature-extractor-evaluator/17254/6)
"""
