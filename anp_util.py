import os
import argparse
import json
from typing import Dict, Union
import warnings

import numpy as np
import torch
from torch.nn import Module, DataParallel
from torchvision.utils import make_grid
from PIL import Image
from tqdm import tqdm

from diffusers import get_cosine_schedule_with_warmup

from anp_model import PerturbConv2d, PerturbLinear
from model import DiffuserModelSched
from anp_config import Config

from util import Log
from dataset import DatasetLoader, Backdoor

from accelerate import Accelerator

def write_json(content: Dict, config: argparse.Namespace, file: str):
    with open(os.path.join(config.output_dir, file), "w") as f:
        return json.dump(content, f, indent=2)

def get_grid_size(n: int):
    if n <= 0:
        return 0, 0
    
    sqrt_min = int(np.floor(np.sqrt(n)))
    for r in reversed(range(sqrt_min + 1)):
        if n % r == 0:
            c = n // r
            return r, c
    return r, c

def auto_rows_cols(n: int, rows: int=None, cols: int=None):
    n = int(n)
    if rows == None and cols != None:
        return n // int(cols), int(cols)
    elif rows != None and cols == None:
        return int(rows), n // int(rows)
    elif rows == None and cols == None:
        return get_grid_size(n=int(n))
    return int(rows), int(cols)

def make_grid(images, rows: int=None, cols: int=None):
    n = len(images)
    rows, cols = auto_rows_cols(n=n, rows=rows, cols=cols)
        
    w, h = images[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    for i, image in enumerate(images):
        grid.paste(image, box=(i%cols*w, i//cols*h))
    return grid

def convert_model(model: Module):
    def replace_bn(module, name):
        '''
        Recursively put desired batch norm in nn.module module.

        set module = net to start code.
        How to replace all ReLU activations in a pretrained network? https://discuss.pytorch.org/t/how-to-replace-all-relu-activations-in-a-pretrained-network/31591/12
        How are the pytorch dimensions for linear layers calculated? https://stackoverflow.com/questions/53784998/how-are-the-pytorch-dimensions-for-linear-layers-calculated
        torch.nn.LazyLinear: https://pytorch.org/docs/stable/generated/torch.nn.LazyLinear.html
        Check the norm of gradients: https://discuss.pytorch.org/t/check-the-norm-of-gradients/27961
        How does BatchNorm keeps track of running_mean?: https://discuss.pytorch.org/t/how-does-batchnorm-keeps-track-of-running-mean/40084/15
        '''
        # go through all attributes of module nn.module (e.g. network or layer) and put batch norms if present
        for attr_str in dir(module):
            target_attr = getattr(module, attr_str)
            # if type(target_attr) == torch.nn.Linear:
            #     print('replaced: ', name, attr_str)
            #     new_lin = PerturbLinear(layer=target_attr)
            #     setattr(module, attr_str, new_lin)
            if type(target_attr) == torch.nn.Conv2d:
                # print('replaced: ', name, attr_str)
                new_conv2d = PerturbConv2d(layer=target_attr)
                setattr(module, attr_str, new_conv2d)

        # iterate through immediate child modules. Note, the recursion is done by our code no need to use named_modules()
        for name, immediate_child_module in module.named_children():
            replace_bn(immediate_child_module, name)
    replace_bn(module=model, name='model')
    return model

def dfs_freeze(model: Module):
    for name, child in model.named_children():
        for param in child.parameters():
            param.requires_grad = False
        dfs_freeze(child)
    return model

def freeze(model: Module):
    for child in model.children():
        for param in child.parameters():
            param.requires_grad = False
    return model

def diff_output(model: Module, perturb_model: Module, noise_sched):
    sample_n = 7
    images = torch.randn((sample_n, 3, 32, 32)).cuda()
    timesteps = torch.randint(0, noise_sched.num_train_timesteps, (sample_n,)).long().cuda()
    # print(f"images shape: {images.shape} | timesteps shape: {timesteps.shape}")

    model = model.cuda()
    with torch.no_grad():
        out0 = model(images, timesteps).sample
        
    perturb_model = perturb_model.cuda()
    with torch.no_grad():
        out1 = perturb_model(images, timesteps).sample
        
    if torch.all(torch.eq(out0, out1)):
        print("Same")
    else:
        print("Different")

def get_model_optim_sched(config: Config, dataset_loader: DatasetLoader):
    model, noise_sched = DiffuserModelSched.get_pretrained(ckpt=config.ckpt, clip_sample=config.clip)
    model = freeze(model=model)
    perturb_model = convert_model(model=model)
    
    diff_output(model=model, perturb_model=perturb_model, noise_sched=noise_sched)
    
    # model = DataParallel(model, device_ids=config.device_ids)
    perturb_model = DataParallel(perturb_model, device_ids=config.device_ids)
    
    parameters = list(perturb_model.named_parameters())
    perturb_params = [v for n, v in parameters if "bn" in n]
    optim = torch.optim.Adam(perturb_params, lr=config.learning_rate)
    
    lr_sched = None
    if config.is_lr_sched:
        lr_sched = get_cosine_schedule_with_warmup(
            optimizer=optim,
            num_warmup_steps=config.lr_warmup_steps,
            num_training_steps=(dataset_loader.num_batch * config.epoch),
        )
    
    return model, perturb_model, optim, noise_sched, lr_sched

def get_data_loader(config: Config):
    ds_root = os.path.join(config.dataset_path)
    # dsl = DatasetLoader(root=ds_root, name=config.dataset, batch_size=config.batch).set_poison(trigger_type=Backdoor.TRIGGER_NONE, target_type=Backdoor.TARGET_TG, clean_rate=1, poison_rate=0).prepare_dataset(mode=DatasetLoader.MODE_FIXED)
    dsl = DatasetLoader(root=ds_root, name=config.dataset, batch_size=config.batch).set_poison(trigger_type=config.trigger, target_type=config.target, clean_rate=0, poison_rate=1).prepare_dataset(mode=DatasetLoader.MODE_FIXED)
    # backdoor_dsl = DatasetLoader(root=ds_root, name=config.dataset, batch_size=config.batch).set_poison(trigger_type=config.trigger, target_type=config.target, clean_rate=0, poison_rate=1).prepare_dataset(mode=DatasetLoader.MODE_FIXED)
    print(f"datasetloader len: {len(dsl)}")
    # return dsl, dsl.get_dataloader(), backdoor_dsl, backdoor_dsl.get_dataloader()
    return dsl, dsl.get_dataloader(), None, None

def get_accelerator(config: Config):
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps, 
        log_with=["tensorboard", "wandb"],
        # log_with="tensorboard",
        logging_dir=os.path.join(config.output_dir, "logs")
    )
    return accelerator

def init_tracker(config: Config, accelerator: Accelerator):
    tracked_config = {}
    for key, val in config.__dict__.items():
        if isinstance(val, int) or isinstance(val, float) or isinstance(val, str) or isinstance(val, bool) or isinstance(val, torch.Tensor):
            tracked_config[key] = val
    accelerator.init_trackers(config.project, config=tracked_config)

def sampling(config: Config, file_name: Union[int, str], pipeline):
    def gen_samples(init: torch.Tensor, folder: Union[os.PathLike, str]):
        test_dir = os.path.join(config.output_dir, folder)
        os.makedirs(test_dir, exist_ok=True)
        
        # Sample some images from random noise (this is the backward diffusion process).
        # The default pipeline output type is `List[PIL.Image]`
        pipline_res = pipeline(
            batch_size = config.eval_sample_n, 
            generator=torch.manual_seed(config.seed),
            init=init,
            output_type=None
        )
        images = pipline_res.images
        movie = pipline_res.movie
        
        # # Because PIL can only accept 2D matrix for gray-scale images, thus, we need to convert the 3D tensors into 2D ones.
        images = [Image.fromarray(image) for image in np.squeeze((images * 255).round().astype("uint8"))]
        init_images = [Image.fromarray(image) for image in np.squeeze((movie[0] * 255).round().astype("uint8"))]

        # # Make a grid out of the images
        image_grid = make_grid(images)
        init_image_grid = make_grid(init_images)

        # sam_obj = Samples(samples=np.array(movie), save_dir=test_dir)
        
        clip_opt = "" if config.clip else "_noclip"
        # # Save the images
        if isinstance(file_name, int):
            image_grid.save(f"{test_dir}/{file_name:04d}{clip_opt}.png")
            init_image_grid.save(f"{test_dir}/{file_name:04d}{clip_opt}_sample_t0.png")
            # sam_obj.save(file_path=f"{file_name:04d}{clip_opt}_samples.pkl")
            # sam_obj.plot_series(slice_idx=slice(None), end_point=True, prefix_img_name=f"{file_name:04d}{clip_opt}_sample_t", animate_name=f"{file_name:04d}{clip_opt}_movie", save_mode=Samples.SAVE_FIRST_LAST, show_mode=Samples.SHOW_NONE)
        elif isinstance(file_name, str):
            image_grid.save(f"{test_dir}/{file_name}{clip_opt}.png")
            init_image_grid.save(f"{test_dir}/{file_name}{clip_opt}_sample_t0.png")
            # sam_obj.save(file_path=f"{file_name}{clip_opt}_samples.pkl")
            # sam_obj.plot_series(slice_idx=slice(None), end_point=True, prefix_img_name=f"{file_name}{clip_opt}_sample_t", animate_name=f"{file_name}{clip_opt}_movie", save_mode=Samples.SAVE_FIRST_LAST, show_mode=Samples.SHOW_NONE)
        else:
            raise TypeError(f"Argument 'file_name' should be string nor integer.")
    
    with torch.no_grad():
        noise = torch.randn(
                    (config.eval_sample_n, pipeline.unet.in_channels, pipeline.unet.sample_size, pipeline.unet.sample_size),
                    generator=torch.manual_seed(config.seed),
                )
        # Sample Clean Samples
        gen_samples(init=noise, folder="samples")
        # Sample Backdoor Samples
        # init = noise + torch.where(dsl.trigger.unsqueeze(0) == -1.0, 0, 1)
        # init = noise + dsl.trigger.unsqueeze(0)
        # print(f"Trigger - (max: {torch.max(dsl.trigger)}, min: {torch.min(dsl.trigger)}) | Noise - (max: {torch.max(noise)}, min: {torch.min(noise)}) | Init - (max: {torch.max(init)}, min: {torch.min(init)})")
        # gen_samples(init=init, folder="backdoor_samples")
        
def save_imgs(imgs: np.ndarray, file_dir: Union[str, os.PathLike], file_name: Union[str, os.PathLike]="") -> None:
    os.makedirs(file_dir, exist_ok=True)
    # Because PIL can only accept 2D matrix for gray-scale images, thus, we need to convert the 3D tensors into 2D ones.
    images = [Image.fromarray(image) for image in np.squeeze((imgs * 255).round().astype("uint8"))]
    for i, img in enumerate(tqdm(images)):
        img.save(os.path.join(file_dir, f"{file_name}{i}.png"))
        
def update_score_file(config: Config, mse_sc: float, ssim_sc: float, epoch: int=None) -> Dict:
    def get_key(config: Config, key):
        res = f"{key}_ep{epoch}" if epoch != None else key
        res += "_noclip" if not config.clip else ""
        return res
    
    def update_dict(data: Dict, key: str, val):
        if val != None:
            data[str(key)] = val
        return data
    
    def update_best(data: Dict, key: str, val, comp_fn):
        key_used = f"{str(key)}_best"
        if val != None:
            if key_used in data:
                data[key_used] = comp_fn([val, data[key_used]])
            else:
                data[key_used] = val
        return data
        
    sc: Dict = {}
    try:
        with open(os.path.join(config.output_dir, config.score_file), "r") as f:
            sc = json.load(f)
    except:
        warnings.warn(Log.info(f"No existed {config.score_file}, create new one"))
    finally:
        with open(os.path.join(config.output_dir, config.score_file), "w") as f:
            # sc = update_dict(data=sc, key=get_key(config=config, key="FID"), val=fid_sc)
            sc = update_dict(data=sc, key=get_key(config=config, key="MSE"), val=mse_sc)
            sc = update_dict(data=sc, key=get_key(config=config, key="SSIM"), val=ssim_sc)
            sc = update_best(data=sc, key="MSE", val=mse_sc, comp_fn=min)
            sc = update_best(data=sc, key="SSIM", val=ssim_sc, comp_fn=max)
            # sc["FID"] = float(fid_sc) if fid_sc != None else sc["FID"]
            # sc["MSE"] = float(mse_sc) if mse_sc != None else sc["MSE"]
            # sc["SSIM"] = float(ssim_sc) if ssim_sc != None else sc["SSIM"]
            json.dump(sc, f, indent=2, sort_keys=True)
        return sc
    
def log_score(config: Config, accelerator: Accelerator, scores: Dict, step: int):    
    def parse_ep(key):
        ep_str = ''.join(filter(str.isdigit, key))
        return config.epoch - 1 if ep_str == '' else int(ep_str)
    
    def parse_clip(key):
        return False if "noclip" in key else True
    
    def parse_metric(key):
        return key.split('_')[0]
    
    def get_log_key(key):
        res = parse_metric(key)
        res += "_noclip" if not parse_clip(key) else ""
        return res
        
    def get_log_ep(key):
        return parse_ep(key)
    
    # accelerator.log({get_key(config=config, key="FID"): fid_sc, "epoch": config.sample_ep})
    # accelerator.log({get_key(config=config, key="MSE"): mse_sc, "epoch": config.sample_ep})
    # accelerator.log({get_key(config=config, key="SSIM"): ssim_sc, "epoch": config.sample_ep})
    
    for key, val in scores.items():
        if 'best' not in key:
            print(f"Log: ({get_log_key(key)}: {val}, epoch: {get_log_ep(key)}, step: {step})")
            accelerator.log({get_log_key(key): val, 'epoch': get_log_ep(key)}, step=step)
        else:
            print(f"Log: ({key}: {val})")
        
    accelerator.log(scores)