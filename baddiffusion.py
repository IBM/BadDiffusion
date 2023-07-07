from dataclasses import dataclass
import argparse
import os
import json
import traceback
from typing import Dict, Union
import warnings

import torch
import wandb

from dataset import DatasetLoader, Backdoor, ImagePathDataset
from fid_score import fid
from util import Log

MODE_TRAIN: str = 'train'
MODE_RESUME: str = 'resume'
MODE_SAMPLING: str = 'sampling'
MODE_MEASURE: str = 'measure'
MODE_TRAIN_MEASURE: str = 'train+measure'

DEFAULT_PROJECT: str = "Default"
DEFAULT_BATCH: int = 512
DEFAULT_EVAL_MAX_BATCH: int = 256
DEFAULT_EPOCH: int = 50
DEFAULT_LEARNING_RATE: float = None
DEFAULT_LEARNING_RATE_32: float = 2e-4
DEFAULT_LEARNING_RATE_256: float = 8e-5
DEFAULT_CLEAN_RATE: float = 1.0
DEFAULT_POISON_RATE: float = 0.007
DEFAULT_TRIGGER: str = Backdoor.TRIGGER_BOX_14
DEFAULT_TARGET: str = Backdoor.TARGET_CORNER
DEFAULT_DATASET_LOAD_MODE: str = DatasetLoader.MODE_FIXED
DEFAULT_GPU = '0'
DEFAULT_CKPT: str = None
DEFAULT_OVERWRITE: bool = False
DEFAULT_POSTFIX: str = ""
DEFAULT_FCLIP: str = 'o'
DEFAULT_SAVE_IMAGE_EPOCHS: int = 20
DEFAULT_SAVE_MODEL_EPOCHS: int = 5
DEFAULT_IS_SAVE_ALL_MODEL_EPOCHS: bool = False
DEFAULT_SAMPLE_EPOCH: int = None
DEFAULT_RESULT: int = '.'

NOT_MODE_TRAIN_OPTS = ['sample_ep']
NOT_MODE_TRAIN_MEASURE_OPTS = ['sample_ep']
MODE_RESUME_OPTS = ['project', 'mode', 'gpu', 'ckpt']
MODE_SAMPLING_OPTS = ['project', 'mode', 'eval_max_batch', 'gpu', 'fclip', 'ckpt', 'sample_ep', 'sched']
MODE_MEASURE_OPTS = ['project', 'mode', 'eval_max_batch', 'gpu', 'fclip', 'ckpt', 'sample_ep', 'sched']
# IGNORE_ARGS = ['overwrite']
IGNORE_ARGS = ['overwrite', 'is_save_all_model_epochs']

def parse_args():
    parser = argparse.ArgumentParser(description=globals()['__doc__'])

    parser.add_argument('--project', '-pj', required=False, type=str, help='Project name')
    parser.add_argument('--mode', '-m', required=True, type=str, help='Train or test the model', choices=[MODE_TRAIN, MODE_RESUME, MODE_SAMPLING, MODE_MEASURE, MODE_TRAIN_MEASURE])
    parser.add_argument('--dataset', '-ds', type=str, help='Training dataset', choices=[DatasetLoader.MNIST, DatasetLoader.CIFAR10, DatasetLoader.CELEBA, DatasetLoader.CELEBA_HQ])
    parser.add_argument('--batch', '-b', type=int, help=f"Batch size, default for train: {DEFAULT_BATCH}")
    parser.add_argument('--sched', '-sc', type=str, help='Noise scheduler', choices=["DDPM-SCHED", "DDIM-SCHED", "DPM_SOLVER_PP_O1-SCHED", "DPM_SOLVER_O1-SCHED", "DPM_SOLVER_PP_O2-SCHED", "DPM_SOLVER_O2-SCHED", "DPM_SOLVER_PP_O3-SCHED", "DPM_SOLVER_O3-SCHED", "UNIPC-SCHED", "PNDM-SCHED", "DEIS-SCHED", "HEUN-SCHED", "SCORE-SDE-VE-SCHED"])
    parser.add_argument('--eval_max_batch', '-eb', type=int, help=f"Batch size of sampling, default for train: {DEFAULT_EVAL_MAX_BATCH}")
    parser.add_argument('--epoch', '-e', type=int, help=f"Epoch num, default for train: {DEFAULT_EPOCH}")
    parser.add_argument('--learning_rate', '-lr', type=float, help=f"Learning rate, default for 32 * 32 image: {DEFAULT_LEARNING_RATE_32}, default for larger images: {DEFAULT_LEARNING_RATE_256}")
    parser.add_argument('--clean_rate', '-cr', type=float, help=f"Clean rate, default for train: {DEFAULT_CLEAN_RATE}")
    parser.add_argument('--poison_rate', '-pr', type=float, help=f"Poison rate, default for train: {DEFAULT_POISON_RATE}")
    parser.add_argument('--trigger', '-tr', type=str, help=f"Trigger pattern, default for train: {DEFAULT_TRIGGER}")
    parser.add_argument('--target', '-ta', type=str, help=f"Target pattern, default for train: {DEFAULT_TARGET}")
    parser.add_argument('--dataset_load_mode', '-dlm', type=str, help=f"Mode of loading dataset, default for train: {DEFAULT_DATASET_LOAD_MODE}", choices=[DatasetLoader.MODE_FIXED, DatasetLoader.MODE_FLEX])
    parser.add_argument('--gpu', '-g', type=str, help=f"GPU usage, default for train/resume: {DEFAULT_GPU}")
    parser.add_argument('--ckpt', '-c', type=str, help=f"Load from the checkpoint, default: {DEFAULT_CKPT}")
    parser.add_argument('--overwrite', '-o', action='store_true', help=f"Overwrite the existed training result or not, default for train/resume: {DEFAULT_CKPT}")
    parser.add_argument('--postfix', '-p', type=str, help=f"Postfix of the name of the result folder, default for train/resume: {DEFAULT_POSTFIX}")
    parser.add_argument('--fclip', '-fc', type=str, help=f"Force to clip in each step or not during sampling/measure, default for train/resume: {DEFAULT_FCLIP}", choices=['w', 'o'])
    parser.add_argument('--save_image_epochs', '-sie', type=int, help=f"Save sampled image per epochs, default: {DEFAULT_SAVE_IMAGE_EPOCHS}")
    parser.add_argument('--save_model_epochs', '-sme', type=int, help=f"Save model per epochs, default: {DEFAULT_SAVE_MODEL_EPOCHS}")
    parser.add_argument('--is_save_all_model_epochs', '-isame', action='store_true', help=f"")
    parser.add_argument('--sample_ep', '-se', type=int, help=f"Select i-th epoch to sample/measure, if no specify, use the lastest saved model, default: {DEFAULT_SAMPLE_EPOCH}")
    parser.add_argument('--result', '-res', type=str, help=f"Output file path, default: {DEFAULT_RESULT}")

    args = parser.parse_args()
    
    return args

@dataclass
class TrainingConfig:
    project: str = DEFAULT_PROJECT
    batch: int = DEFAULT_BATCH
    epoch: int = DEFAULT_EPOCH
    eval_max_batch: int = DEFAULT_EVAL_MAX_BATCH
    learning_rate: float = DEFAULT_LEARNING_RATE
    clean_rate: float = DEFAULT_CLEAN_RATE
    poison_rate: float = DEFAULT_POISON_RATE
    trigger: str = DEFAULT_TRIGGER
    target: str = DEFAULT_TARGET
    dataset_load_mode: str = DEFAULT_DATASET_LOAD_MODE
    gpu: str = DEFAULT_GPU
    ckpt: str = DEFAULT_CKPT
    overwrite: bool = DEFAULT_OVERWRITE
    postfix: str  = DEFAULT_POSTFIX
    fclip: str = DEFAULT_FCLIP
    save_image_epochs: int = DEFAULT_SAVE_IMAGE_EPOCHS
    save_model_epochs: int = DEFAULT_SAVE_MODEL_EPOCHS
    is_save_all_model_epochs: bool = DEFAULT_IS_SAVE_ALL_MODEL_EPOCHS
    sample_ep: int = DEFAULT_SAMPLE_EPOCH
    result: str = DEFAULT_RESULT
    
    eval_sample_n: int = 16  # how many images to sample during evaluation
    measure_sample_n: int = 16
    batch_32: int = 128
    batch_256: int = 64
    gradient_accumulation_steps: int = 1
    learning_rate_32_scratch: float = 2e-4
    learning_rate_256_scratch: float = 2e-5
    lr_warmup_steps: int = 500
    # save_image_epochs: int = 1
    mixed_precision: str = 'fp16'  # `no` for float32, `fp16` for automatic mixed precision

    push_to_hub: bool = False  # whether to upload the saved model to the HF Hub
    hub_private_repo: bool = False  
    overwrite_output_dir: bool = True  # overwrite the old model when re-running the notebook
    seed: int = 0
    dataset_path: str = 'datasets'
    ckpt_dir: str = 'ckpt'
    data_ckpt_dir: str = 'data.ckpt'
    ep_model_dir: str = 'epochs'
    ckpt_path: str = None
    data_ckpt_path: str = None
    # hub_token = "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"

def naming_fn(config: TrainingConfig):
    add_on: str = ""
    # add_on += "_clip" if config.clip else ""
    add_on += f"_{config.postfix}" if config.postfix else ""
    return f'res_{config.ckpt}_{config.dataset}_ep{config.epoch}_c{config.clean_rate}_p{config.poison_rate}_{config.trigger}-{config.target}{add_on}'

def read_json(args: argparse.Namespace, file: str):
    with open(os.path.join(args.ckpt, file), "r") as f:
        return json.load(f)

def write_json(content: Dict, config: argparse.Namespace, file: str):
    with open(os.path.join(config.output_dir, file), "w") as f:
        return json.dump(content, f, indent=2)

def setup():
    args_file: str = "args.json"
    config_file: str = "config.json"
    sampling_file: str = "sampling.json"
    measure_file: str = "measure.json"
    
    args: argparse.Namespace = parse_args()
    config: TrainingConfig = TrainingConfig()
    args_data: Dict = {}
    
    if args.mode == MODE_RESUME or args.mode == MODE_SAMPLING or args.mode == MODE_MEASURE:
        with open(os.path.join(args.ckpt, args_file), "r") as f:
            args_data = json.load(f)
        
        for key, value in args_data.items():
            if value != None:
                setattr(config, key, value)
        setattr(config, "output_dir", args.ckpt)
    
    for key, value in args.__dict__.items():
        if args.mode == MODE_TRAIN and (key not in NOT_MODE_TRAIN_OPTS) and value != None:
            setattr(config, key, value)
        elif args.mode == MODE_TRAIN_MEASURE and (key not in NOT_MODE_TRAIN_MEASURE_OPTS) and value != None:
            setattr(config, key, value)
        elif args.mode == MODE_RESUME and key in MODE_RESUME_OPTS and value != None:
            setattr(config, key, value)
        elif args.mode == MODE_SAMPLING and key in MODE_SAMPLING_OPTS and value != None:
            setattr(config, key, value)
        elif args.mode == MODE_MEASURE and key in MODE_MEASURE_OPTS and value != None:
            setattr(config, key, value)
        elif value != None and not (key in IGNORE_ARGS):
            raise NotImplementedError(f"Argument: {key}={value} isn't used in mode: {args.mode}")
        
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", config.gpu)

    print(f"PyTorch detected number of availabel devices: {torch.cuda.device_count()}")
    setattr(config, "device_ids", [int(i) for i in range(len(config.gpu.split(',')))])
    
    # sample_ep options
    if isinstance(config.sample_ep, int):
        if config.sample_ep < 0:
            config.sample_ep = None
    
    # Clip option
    if config.fclip == 'w':
        setattr(config, "clip", True)
    elif config.fclip == 'o':
        setattr(config, "clip", False)
    else:
        setattr(config, "clip", None)
        
    # Determine gradient accumulation & Learning Rate
    bs = 0
    if config.dataset in [DatasetLoader.CIFAR10, DatasetLoader.MNIST]:
        bs = config.batch_32
        if config.learning_rate == None:
            if config.ckpt == None:
                config.learning_rate = config.learning_rate_32_scratch
            else:
                config.learning_rate = DEFAULT_LEARNING_RATE_32
    elif config.dataset in [DatasetLoader.CELEBA, DatasetLoader.CELEBA_HQ, DatasetLoader.LSUN_CHURCH, DatasetLoader.LSUN_BEDROOM]:
        bs = config.batch_256
        if config.learning_rate == None:
            if config.ckpt == None:
                config.learning_rate = config.learning_rate_256_scratch
            else:
                config.learning_rate = DEFAULT_LEARNING_RATE_256
    else:
        raise NotImplementedError()
    if bs % config.batch != 0:
        raise ValueError(f"batch size {config.batch} should be divisible to {bs} for dataset {config.dataset}")
    if bs < config.batch:
        raise ValueError(f"batch size {config.batch} should be smaller or equal to {bs} for dataset {config.dataset}")
    config.gradient_accumulation_steps = int(bs // config.batch)
    
    if args.mode == MODE_TRAIN or args.mode == MODE_TRAIN_MEASURE:
        setattr(config, "output_dir", os.path.join(config.result, naming_fn(config=config)))
    
    print(f"MODE: {config.mode}")
    if config.mode == MODE_TRAIN or args.mode == MODE_TRAIN_MEASURE:
        if not config.overwrite and os.path.isdir(config.output_dir):
            raise ValueError(f"Output directory: {config.output_dir} has already been created, please set overwrite flag --overwrite or -o")
        
        os.makedirs(config.output_dir, exist_ok=True)
        
        write_json(content=vars(args), config=config, file=args_file)
        write_json(content=config.__dict__, config=config, file=config_file)
    elif config.mode == MODE_SAMPLING:
        write_json(content=config.__dict__, config=config, file=sampling_file)
    elif config.mode == MODE_MEASURE or args.mode == MODE_TRAIN_MEASURE:
        write_json(content=config.__dict__, config=config, file=measure_file)
    elif config.mode == MODE_RESUME:
        pass
    else:
        raise NotImplementedError(f"Mode: {config.mode} isn't defined")
    
    if config.ckpt_path == None:
        config.ckpt_path = os.path.join(config.output_dir, config.ckpt_dir)
        config.data_ckpt_path = os.path.join(config.output_dir, config.data_ckpt_dir)
        os.makedirs(config.ckpt_path, exist_ok=True)
    
    name_id = str(config.output_dir).split('/')[-1]
    wandb.init(project=config.project, name=name_id, id=name_id, settings=wandb.Settings(start_method="fork"))
    print(f"Argument Final: {config.__dict__}")
    return config

config = setup()
"""## Config

For convenience, we define a configuration grouping all the training hyperparameters. This would be similar to the arguments used for a [training script](https://github.com/huggingface/diffusers/tree/main/examples).
Here we choose reasonable defaults for hyperparameters like `num_epochs`, `learning_rate`, `lr_warmup_steps`, but feel free to adjust them if you train on your own dataset. For example, `num_epochs` can be increased to 100 for better visual quality.
"""

import numpy as np
from PIL import Image
from torch import nn
from torchmetrics import StructuralSimilarityIndexMeasure
from accelerate import Accelerator
# from diffusers.hub_utils import init_git_repo, push_to_hub
from tqdm.auto import tqdm

from diffusers import DDPMPipeline
from diffusers.optimization import get_cosine_schedule_with_warmup

from model import DiffuserModelSched, batch_sampling, batch_sampling_save
# from util import Samples, MemoryLog, match_count
from util import Samples, MemoryLog
from loss import p_losses_diffuser

def get_accelerator(config: TrainingConfig):
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps, 
        log_with=["tensorboard", "wandb"],
        # log_with="tensorboard",
        logging_dir=os.path.join(config.output_dir, "logs")
    )
    return accelerator

def init_tracker(config: TrainingConfig, accelerator: Accelerator):
    tracked_config = {}
    for key, val in config.__dict__.items():
        if isinstance(val, int) or isinstance(val, float) or isinstance(val, str) or isinstance(val, bool) or isinstance(val, torch.Tensor):
            tracked_config[key] = val
    accelerator.init_trackers(config.project, config=tracked_config)

def get_data_loader(config: TrainingConfig):
    ds_root = os.path.join(config.dataset_path)
    dsl = DatasetLoader(root=ds_root, name=config.dataset, batch_size=config.batch).set_poison(trigger_type=config.trigger, target_type=config.target, clean_rate=config.clean_rate, poison_rate=config.poison_rate).prepare_dataset(mode=config.dataset_load_mode)
    # image_size = dsl.image_size
    # channels = dsl.channel
    # dataset = dsl.get_dataset()
    # loader = dsl.get_dataloader()
    print(f"datasetloader len: {len(dsl)}")
    return dsl

def get_repo(config: TrainingConfig, accelerator: Accelerator):
    repo = None
    if accelerator.is_main_process:
        # if config.push_to_hub:
        #     repo = init_git_repo(config, at_init=True)
        # accelerator.init_trackers(config.output_dir, config=config.__dict__)
        init_tracker(config=config, accelerator=accelerator)
    return repo
        
def get_model_optim_sched(config: TrainingConfig, accelerator: Accelerator, dataset_loader: DatasetLoader):
    if config.ckpt != None:
        if config.sample_ep != None and config.mode in [MODE_MEASURE, MODE_SAMPLING]:
            ep_model_path = get_ep_model_path(config=config, dir=config.ckpt, epoch=config.sample_ep)
            model, noise_sched = DiffuserModelSched.get_pretrained(ckpt=ep_model_path, clip_sample=config.clip)
        # else:
        #     model, noise_sched = DiffuserModelSched.get_pretrained(ckpt=config.ckpt, clip_sample=config.clip)
        #     warnings.warn(Log.warning(f"No such pretrained model: {ep_model_path}, load from ckpt: {config.ckpt}"))
        #     print(Log.warning(f"No such pretrained model: {ep_model_path}, load from ckpt: {config.ckpt}"))
        else:
            model, noise_sched, get_pipeline = DiffuserModelSched.get_pretrained(ckpt=config.ckpt, clip_sample=config.clip)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    else:
        model, noise_sched, get_pipeline = DiffuserModelSched.get_model_sched(image_size=dataset_loader.image_size, channels=dataset_loader.channel, model_type=DiffuserModelSched.MODEL_DEFAULT, noise_sched_type=config.sched, clip_sample=config.clip)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
        
    model = nn.DataParallel(model, device_ids=config.device_ids)
        
    lr_sched = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=config.lr_warmup_steps,
        num_training_steps=(dataset_loader.num_batch * config.epoch),
    )
    
    cur_epoch = cur_step = 0
    
    accelerator.register_for_checkpointing(model, optimizer, lr_sched)
    if config.mode == MODE_RESUME:
        if config.ckpt == None:
            raise ValueError(f"Argument 'ckpt' shouldn't be None for resume mode")
        accelerator.load_state(config.ckpt_path)
        data_ckpt = torch.load(config.data_ckpt_path)
        cur_epoch = data_ckpt['epoch']
        cur_step = data_ckpt['step']
    
    return model, optimizer, lr_sched, noise_sched, cur_epoch, cur_step, get_pipeline

def init_train(config: TrainingConfig, dataset_loader: DatasetLoader):
    # Initialize accelerator and tensorboard logging    
    accelerator = get_accelerator(config=config)
    repo = get_repo(config=config, accelerator=accelerator)
    
    model, optimizer, lr_sched, noise_sched, cur_epoch, cur_step, get_pipeline = get_model_optim_sched(config=config, accelerator=accelerator, dataset_loader=dataset_loader)
    
    dataloader = dataset_loader.get_dataloader()
    model, optimizer, dataloader, lr_sched = accelerator.prepare(
        model, optimizer, dataloader, lr_sched
    )
    return accelerator, repo, model, noise_sched, optimizer, dataloader, lr_sched, cur_epoch, cur_step, get_pipeline

def make_grid(images, rows, cols):
    w, h = images[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    for i, image in enumerate(images):
        grid.paste(image, box=(i%cols*w, i//cols*h))
    return grid

def sampling(config: TrainingConfig, file_name: Union[int, str], pipeline):
    def gen_samples(init: torch.Tensor, folder: Union[os.PathLike, str]):
        test_dir = os.path.join(config.output_dir, folder)
        os.makedirs(test_dir, exist_ok=True)
        
        # Sample some images from random noise (this is the backward diffusion process).
        # The default pipeline output type is `List[PIL.Image]`
        pipline_res = pipeline(
            batch_size = config.eval_sample_n, 
            generator=torch.manual_seed(config.seed),
            init=init,
            output_type=None,
            save_every_step=True
        )
        images = pipline_res.images
        movie = pipline_res.movie
        
        # # Because PIL can only accept 2D matrix for gray-scale images, thus, we need to convert the 3D tensors into 2D ones.
        images = [Image.fromarray(image) for image in np.squeeze((images * 255).round().astype("uint8"))]
        init_images = [Image.fromarray(image) for image in np.squeeze((movie[0] * 255).round().astype("uint8"))]

        # # Make a grid out of the images
        image_grid = make_grid(images, rows=4, cols=4)
        init_image_grid = make_grid(init_images, rows=4, cols=4)

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
        init = noise + dsl.trigger.unsqueeze(0)
        # print(f"Trigger - (max: {torch.max(dsl.trigger)}, min: {torch.min(dsl.trigger)}) | Noise - (max: {torch.max(noise)}, min: {torch.min(noise)}) | Init - (max: {torch.max(init)}, min: {torch.min(init)})")
        gen_samples(init=init, folder="backdoor_samples")

def save_imgs(imgs: np.ndarray, file_dir: Union[str, os.PathLike], file_name: Union[str, os.PathLike]="") -> None:
    os.makedirs(file_dir, exist_ok=True)
    # Because PIL can only accept 2D matrix for gray-scale images, thus, we need to convert the 3D tensors into 2D ones.
    images = [Image.fromarray(image) for image in np.squeeze((imgs * 255).round().astype("uint8"))]
    for i, img in enumerate(tqdm(images)):
        img.save(os.path.join(file_dir, f"{file_name}{i}.png"))

def update_score_file(config: TrainingConfig, score_file: str, fid_sc: float, mse_sc: float, ssim_sc: float) -> Dict:
    def get_key(config: TrainingConfig, key):
        res = f"{key}_ep{config.sample_ep}" if config.sample_ep != None else key
        res += "_noclip" if not config.clip else ""
        return res
    
    def update_dict(data: Dict, key: str, val):
        data[str(key)] = val if val != None else data[str(key)]
        return data
        
    sc: Dict = {}
    try:
        with open(os.path.join(config.output_dir, score_file), "r") as f:
            sc = json.load(f)
    except:
        Log.info(f"No existed {score_file}, create new one")
    finally:
        with open(os.path.join(config.output_dir, score_file), "w") as f:
            sc = update_dict(data=sc, key=get_key(config=config, key="FID"), val=fid_sc)
            sc = update_dict(data=sc, key=get_key(config=config, key="MSE"), val=mse_sc)
            sc = update_dict(data=sc, key=get_key(config=config, key="SSIM"), val=ssim_sc)
            json.dump(sc, f, indent=2, sort_keys=True)
        return sc
    
def log_score(config: TrainingConfig, accelerator: Accelerator, scores: Dict, step: int):    
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
    
    for key, val in scores.items():
        print(f"Log: ({get_log_key(key)}: {val}, epoch: {get_log_ep(key)}, step: {step})")
        accelerator.log({get_log_key(key): val, 'epoch': get_log_ep(key)}, step=step)
        
    accelerator.log(scores)

def measure(config: TrainingConfig, accelerator: Accelerator, dataset_loader: DatasetLoader, folder_name: Union[int, str], pipeline, resample: bool=True, recomp: bool=True):
    score_file = "score.json"
    
    fid_sc = mse_sc = ssim_sc = None
    re_comp_clean_metric = False
    re_comp_backdoor_metric = False
    
    # Random Number Generator
    rng = torch.Generator()
    rng.manual_seed(config.seed)
    
    # Dataset samples
    ds = dataset_loader.get_dataset().shuffle(seed=config.seed)
    step = dataset_loader.num_batch * (config.sample_ep + 1 if config.sample_ep != None else config.epoch)
    
    # Folders
    dataset_img_dir = os.path.join(folder_name, config.dataset)
    folder_path_ls = [config.output_dir, folder_name]
    if config.sample_ep != None:
        folder_path_ls += [f"ep{config.sample_ep}"]
    clean_folder = "clean" + ("_noclip" if not config.clip else "")
    backdoor_folder = "backdoor" + ("_noclip" if not config.clip else "")
    clean_path = os.path.join(*folder_path_ls, clean_folder)
    backdoor_path = os.path.join(*folder_path_ls, backdoor_folder)
    
    # if not os.path.isdir(dataset_img_dir) or resample:
    if not os.path.isdir(dataset_img_dir):
        os.makedirs(dataset_img_dir, exist_ok=True)
        # dataset_loader.show_sample(img=ds[0][DatasetLoader.IMAGE], is_show=False, file_name=os.path.join(clean_measure_dir, f"0.png"))
        for i, img in enumerate(tqdm(ds[:config.measure_sample_n][DatasetLoader.IMAGE])):
            dataset_loader.show_sample(img=img, is_show=False, file_name=os.path.join(dataset_img_dir, f"{i}.png"))
        re_comp_clean_metric = True
    
    # Init noise
    noise = torch.randn(
                (config.measure_sample_n, pipeline.unet.in_channels, pipeline.unet.sample_size, pipeline.unet.sample_size),
                generator=torch.manual_seed(config.seed),
            )
    backdoor_noise = noise + dataset_loader.trigger.unsqueeze(0)
    
    # Sampling
    # if not os.path.isdir(clean_dir) or match_count(dir=clean_dir) < config.measure_sample_n or resample:
    if not os.path.isdir(clean_path) or resample:
    # clean_sample_imgs = batch_sampling(sample_n=config.measure_sample_n, pipeline=pipeline, init=noise, max_batch_n=config.eval_max_batch, rng=rng)
    # save_imgs(imgs=clean_sample_imgs, file_dir=clean_path, file_name="")
        batch_sampling_save(sample_n=config.measure_sample_n, pipeline=pipeline, path=clean_path, init=noise, max_batch_n=config.eval_max_batch, rng=rng)
        re_comp_clean_metric = True
    # if not os.path.isdir(backdoor_dir) or match_count(dir=backdoor_dir) < config.measure_sample_n or resample:
    if not os.path.isdir(backdoor_path) or resample:
    # backdoor_sample_imgs = batch_sampling(sample_n=config.measure_sample_n, pipeline=pipeline, init=backdoor_noise, max_batch_n=config.eval_max_batch, rng=rng)
    # save_imgs(imgs=backdoor_sample_imgs, file_dir=backdoor_path, file_name="")
        batch_sampling_save(sample_n=config.measure_sample_n, pipeline=pipeline, path=backdoor_path, init=backdoor_noise,  max_batch_n=config.eval_max_batch, rng=rng)
        re_comp_backdoor_metric = True
    
    # Compute Score
    if re_comp_clean_metric or recomp:
        fid_sc = float(fid(path=[dataset_img_dir, clean_path], device=config.device_ids[0], num_workers=4))
    
    if re_comp_backdoor_metric or recomp:
        device = torch.device(config.device_ids[0])
        # gen_backdoor_target = torch.from_numpy(backdoor_sample_imgs)
        # print(f"backdoor_sample_imgs shape: {backdoor_sample_imgs.shape}")
        gen_backdoor_target = ImagePathDataset(path=backdoor_path)[:].to(device)
        
        reps = ([len(gen_backdoor_target)] + ([1] * (len(dsl.target.shape))))
        backdoor_target = torch.squeeze((dsl.target.repeat(*reps) / 2 + 0.5).clamp(0, 1)).to(device)
        
        print(f"gen_backdoor_target: {gen_backdoor_target.shape}, vmax: {torch.max(gen_backdoor_target)}, vmin: {torch.min(backdoor_target)} | backdoor_target: {backdoor_target.shape}, vmax: {torch.max(backdoor_target)}, vmin: {torch.min(backdoor_target)}")
        mse_sc = float(nn.MSELoss(reduction='mean')(gen_backdoor_target, backdoor_target))
        ssim_sc = float(StructuralSimilarityIndexMeasure(data_range=1.0).to(device)(gen_backdoor_target, backdoor_target))
    print(f"[{config.sample_ep}] FID: {fid_sc}, MSE: {mse_sc}, SSIM: {ssim_sc}")
    
    sc = update_score_file(config=config, score_file=score_file, fid_sc=fid_sc, mse_sc=mse_sc, ssim_sc=ssim_sc)
    # accelerator.log(sc)
    log_score(config=config, accelerator=accelerator, scores=sc, step=step)

"""With this in end, we can group all together and write our training function. This just wraps the training step we saw in the previous section in a loop, using Accelerate for easy TensorBoard logging, gradient accumulation, mixed precision training and multi-GPUs or TPU training."""

def get_ep_model_path(config: TrainingConfig, dir: Union[str, os.PathLike], epoch: int):
    return os.path.join(dir, config.ep_model_dir, f"ep{epoch}")

def checkpoint(config: TrainingConfig, accelerator: Accelerator, pipeline, cur_epoch: int, cur_step: int, repo=None, commit_msg: str=None):
    accelerator.save_state(config.ckpt_path)
    accelerator.save({'epoch': cur_epoch, 'step': cur_step}, config.data_ckpt_path)
    # if config.push_to_hub:
    #     push_to_hub(config, pipeline, repo, commit_message=commit_msg, blocking=True)
    # else:
    pipeline.save_pretrained(config.output_dir)
        
    if config.is_save_all_model_epochs:
        # ep_model_path = os.path.join(config.output_dir, config.ep_model_dir, f"ep{cur_epoch}")
        ep_model_path = get_ep_model_path(config=config, dir=config.output_dir, epoch=cur_epoch)
        os.makedirs(ep_model_path, exist_ok=True)
        pipeline.save_pretrained(ep_model_path)

def train_loop(config: TrainingConfig, accelerator: Accelerator, repo, model: nn.Module, get_pipeline, noise_sched, optimizer: torch.optim, loader, lr_sched, start_epoch: int=0, start_step: int=0):
    try:
        # memlog = MemoryLog('memlog.log')
        cur_step = start_step
        epoch = start_epoch
        
        # Test evaluate
        # memlog.append()
        # pipeline = DDPMPipeline(unet=accelerator.unwrap_model(model), scheduler=noise_sched)        
        pipeline = get_pipeline(unet=accelerator.unwrap_model(model), scheduler=noise_sched)
        sampling(config, 0, pipeline)
        # memlog.append()

        # Now you train the model
        for epoch in range(int(start_epoch), int(config.epoch)):
            progress_bar = tqdm(total=len(loader), disable=not accelerator.is_local_main_process)
            progress_bar.set_description(f"Epoch {epoch}")

            for step, batch in enumerate(loader):
                # memlog.append()
                # clean_images = batch['images']
                clean_images = batch['pixel_values'].to(model.device_ids[0])
                target_images = batch["target"].to(model.device_ids[0])
                # Sample noise to add to the images
                noise = torch.randn(clean_images.shape).to(clean_images.device)
                bs = clean_images.shape[0]

                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_sched.num_train_timesteps, (bs,), device=clean_images.device).long()

                # Add noise to the clean images according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                
                with accelerator.accumulate(model):
                    # Predict the noise residual
                    loss = p_losses_diffuser(noise_sched, model=model, x_start=target_images, R=clean_images, timesteps=timesteps, noise=noise, loss_type="l2")
                    accelerator.backward(loss)
                    
                    # clip_grad_norm_: https://huggingface.co/docs/accelerate/v0.13.2/en/package_reference/accelerator#accelerate.Accelerator.clip_grad_norm_
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    lr_sched.step()
                    optimizer.zero_grad()
                # memlog.append()
                
                progress_bar.update(1)
                logs = {"loss": loss.detach().item(), "lr": lr_sched.get_last_lr()[0], "epoch": epoch, "step": cur_step}
                progress_bar.set_postfix(**logs)
                accelerator.log(logs, step=cur_step)
                cur_step += 1

            # After each epoch you optionally sample some demo images with evaluate() and save the model
            if accelerator.is_main_process:
                # pipeline = DDPMPipeline(unet=accelerator.unwrap_model(model), scheduler=noise_sched)
                pipeline = get_pipeline(unet=accelerator.unwrap_model(model), scheduler=noise_sched)

                if (epoch + 1) % config.save_image_epochs == 0 or epoch == config.epoch - 1:
                    sampling(config, epoch, pipeline)

                if (epoch + 1) % config.save_model_epochs == 0 or epoch == config.epoch - 1:
                    checkpoint(config=config, accelerator=accelerator, pipeline=pipeline, cur_epoch=epoch, cur_step=cur_step, repo=repo, commit_msg=f"Epoch {epoch}")
            # memlog.append()
    except:
        Log.error("Training process is interrupted by an error")
        print(traceback.format_exc())
    finally:
        Log.info("Save model and sample images")
        # pipeline = DDPMPipeline(unet=accelerator.unwrap_model(model), scheduler=noise_sched)        
        pipeline = get_pipeline(unet=accelerator.unwrap_model(model), scheduler=noise_sched)
        if accelerator.is_main_process:
            checkpoint(config=config, accelerator=accelerator, pipeline=pipeline, cur_epoch=epoch, cur_step=cur_step, repo=repo, commit_msg=f"Epoch {epoch}")
            sampling(config, 'final', pipeline)
        return pipeline

"""## Let's train!

Let's launch the training (including multi-GPU training) from the notebook using Accelerate's `notebook_launcher` function:
"""
dsl = get_data_loader(config=config)
accelerator, repo, model, noise_sched, optimizer, dataloader, lr_sched, cur_epoch, cur_step, get_pipeline = init_train(config=config, dataset_loader=dsl)

if config.mode == MODE_TRAIN or config.mode == MODE_RESUME or config.mode == MODE_TRAIN_MEASURE:
    pipeline = train_loop(config, accelerator, repo, model, get_pipeline, noise_sched, optimizer, dataloader, lr_sched, start_epoch=cur_epoch, start_step=cur_step)

    if config.mode == MODE_TRAIN_MEASURE and accelerator.is_main_process:
        accelerator.free_memory()
        accelerator.clear()
        measure(config=config, accelerator=accelerator, dataset_loader=dsl, folder_name='measure', pipeline=pipeline)
elif config.mode == MODE_SAMPLING:
    # pipeline = DDPMPipeline(unet=accelerator.unwrap_model(model), scheduler=noise_sched)
    pipeline = get_pipeline(unet=accelerator.unwrap_model(model), scheduler=noise_sched)
    if config.sample_ep != None:
        sampling(config=config, file_name=int(config.sample_ep), pipeline=pipeline)
    else:
        sampling(config=config, file_name="final", pipeline=pipeline)
elif config.mode == MODE_MEASURE:
    # pipeline = DDPMPipeline(unet=accelerator.unwrap_model(model), scheduler=noise_sched)
    pipeline = get_pipeline(unet=accelerator.unwrap_model(model), scheduler=noise_sched)
    measure(config=config, accelerator=accelerator, dataset_loader=dsl, folder_name='measure', pipeline=pipeline)
    if config.sample_ep != None:
        sampling(config=config, file_name=int(config.sample_ep), pipeline=pipeline)
    else:
        sampling(config=config, file_name="final", pipeline=pipeline)
else:
    raise NotImplementedError()

accelerator.end_training()
