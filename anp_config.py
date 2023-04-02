import os
import json
import argparse
from dataclasses import dataclass, field
from typing import Dict, List, Union
import wandb

from dataset import Backdoor

@dataclass    
class Config:
    project: str = "anp_test"
    dataset_path: Union[str, os.PathLike] = "datasets"
    dataset: str = "CIFAR10"
    batch: int = 128
    epoch: int = 10
    trigger: str = Backdoor.TRIGGER_NONE 
    target: str = Backdoor.TARGET_TG
    poison_date: float = None
    ckpt: Union[str, os.PathLike] = None
    clip: bool = True
    learning_rate: float = 1e-4
    momentum: float = 0.9
    is_lr_sched: bool = False
    clip: bool = True
    gpu: str = '0'
    perturb_budget: float = 4.0
    tag: str = None
    
    measure_sample_n: int = 128
    eval_sample_n: int = 16
    save_image_epochs: int = 1
    save_model_epochs: int = 5
    
    output_dir: Union[str, os.PathLike] = ""
    measure_dir: Union[str, os.PathLike] = "measure"
    score_file: Union[str, os.PathLike] = "score.json"
    gradient_accumulation_steps: int = 1
    lr_warmup_steps: int = 500
    mixed_precision: str = "fp16"
    seed: int = 0
    device_ids: List[int] = field(default_factory=lambda: [0])

def write_json(content: Dict, config: argparse.Namespace, file: str):
    with open(os.path.join(config.output_dir, file), "w") as f:
        return json.dump(content, f, indent=2)
    
def naming_fn(config: Config):
    add_on = "_sched" if config.lr_sched else ""
    add_on += f"_{config.tag}" if config.tag != None else ""
    return f"res_anp_{config.epoch}_lr{config.learning_rate}_pb{config.perturb_budget}{add_on}_{config.ckpt}"
    
def get_config():
    args_file = "args.json"
    config = Config()
    
    parser = argparse.ArgumentParser(description=globals()['__doc__'])
    
    parser.add_argument('--project', '-pj', type=str, help=f"Project name")
    parser.add_argument('--epoch', '-e', type=int, default=config.epoch, help=f"Training epochs, default: {config.epoch}")
    parser.add_argument('--learning_rate', '-lr', type=float, default=config.learning_rate, help=f"Learning rate, default: {config.learning_rate}")
    parser.add_argument('--lr_sched', '-sch', action='store_true', help=f"Use LR Scheduler")
    parser.add_argument('--perturb_budget', '-pb', type=float, default=config.perturb_budget, help=f"Perturbation budget, default: {config.perturb_budget}")
    parser.add_argument('--output_dir', '-od', type=str, help=f"Output directory")
    parser.add_argument('--tag', '-t', type=str, help=f"Additional information added on the name")
    parser.add_argument('--gpu', '-g', type=str, default=config.gpu, help=f"GPU usage, default for train/resume: {config.gpu}")
    parser.add_argument('--ckpt', '-c', type=str, help=f"Load from the checkpoint")
    
    args = parser.parse_args()
    
    for key, value in args.__dict__.items():
        if value != None:
            setattr(config, key, value)
    if config.output_dir != "" or config.output_dir != None:
        config.output_dir = os.path.join(config.output_dir, naming_fn(config=config))
    else:
        config.output_dir = naming_fn(config=config)
    
    with open(os.path.join(config.ckpt, args_file), "r") as f:
        args_data = json.load(f)
        config.trigger = args_data['trigger']
        config.target = args_data['target']
        config.poison_rate = args_data['poison_rate']
        config.dataset = args_data['dataset']
        config.backdoor_lr = args_data['learning_rate'] if 'learning_rate' in args_data else None
    
    print(config.__dict__)
    
    config.device_ids = [int(i) for i in range(len(config.gpu.split(',')))]
    
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(config.gpu))
    os.environ["CUDA_VISIBLE_DEVICES"] = str(config.gpu)
    os.makedirs(config.output_dir, exist_ok=True)
    write_json(content=config.__dict__, config=config, file="config.json")
    
    name_id = str(config.output_dir).split('/')[-1]
    wandb.init(project=config.project, name=name_id, id=name_id, settings=wandb.Settings(start_method="fork"))
    
    return config