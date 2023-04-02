# %%
from ast import Slice
from dataclasses import dataclass
from math import ceil, floor, sqrt
import os
from datetime import datetime
from typing import Tuple, Union, Dict, List
import warnings

import psutil
from matplotlib import pyplot as plt
import matplotlib.animation as animation
from matplotlib.dates import SA
import numpy as np
import torch
from torchvision.utils import make_grid, save_image
from PIL import Image
from comet_ml import Experiment, ExistingExperiment

# import src.monitor.logger as logger

def match_count(dir: Union[str, os.PathLike], exts: List[str]=["png", "jpg", "jpeg"]) -> int:
    files_grabbed = []
    for ext in exts:
        files_grabbed.extend(glob.glob(os.path.join(dir, f"*.{ext}")))
    return len(set(files_grabbed))
class Log:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    
    @staticmethod
    def error(msg: str):
        return Log.FAIL + Log.BOLD + msg + Log.ENDC
    
    @staticmethod
    def warning(msg: str):
        return Log.WARNING + Log.BOLD + msg + Log.ENDC

    @staticmethod
    def info(msg: str):
        return Log.OKGREEN + Log.BOLD + msg + Log.ENDC

class MemoryLog:
    def __init__(self, file: Union[str, os.PathLike]="mem.log"):
        self.__log_file = file
        self.__f = open(self.__log_file, "a")
        self.__f.writelines("Time, Ram Usage, GPU Mem Usage" + "\n")
    
    @staticmethod
    def mem_infos2str(mem_info):
        return "MEM: {:.2f}%".format(mem_info.percent)
    
    @staticmethod
    def gpu_infos2str(gpu_infos: List[Dict]):
        res = ""
        for i, info in enumerate(gpu_infos):
            res += "[GPU:{:d}] {:.2f}% ".format(i, (1 - (info['free'] / info['all'])) * 100)
        return res
    
    def append(self):
        available_gpus = [i for i in range(torch.cuda.device_count())]
        current_time = datetime.now().strftime("%Y/%m/%d - %H:%M:%S")
        mem_info = psutil.virtual_memory()
        gpu_infos = []
        for gpu_id in available_gpus:
            free_mem, all_mem = torch.cuda.mem_get_info(gpu_id)
            gpu_infos.append({'free': free_mem, 'all': all_mem})
        msg = current_time + ', ' + MemoryLog.mem_infos2str(mem_info) + ', ' + MemoryLog.gpu_infos2str(gpu_infos)
        self.__f.writelines(msg + "\n")
        self.__f.flush()
        
    def __del__(self):
        self.__f.flush()
        self.__f.close()

def normalize(x: Union[np.ndarray, torch.Tensor], vmin_in: float=None, vmax_in: float=None, vmin_out: float=0, vmax_out: float=1, eps: float=1e-5) -> Union[np.ndarray, torch.Tensor]:
    if vmax_out == None and vmin_out == None:
        return x

    if isinstance(x, np.ndarray):
        if vmin_in == None:
            min_x = np.min(x)
        else:
            min_x = vmin_in
        if vmax_in == None:
            max_x = np.max(x)
        else:
            max_x = vmax_in
    elif isinstance(x, torch.Tensor):
        if vmin_in == None:
            min_x = torch.min(x)
        else:
            min_x = vmin_in
        if vmax_in == None:
            max_x = torch.max(x)
        else:
            max_x = vmax_in
    else:
        raise TypeError("x must be a torch.Tensor or a np.ndarray")
    if vmax_out == None:
        vmax_out = max_x
    if vmin_out == None:
        vmin_out = min_x
    return ((x - min_x) / (max_x - min_x + eps)) * (vmax_out - vmin_out) + vmin_out

# @dataclass
# class Image:
#     channel: int
#     channel_loc: int
#     vmin: Union[float, int]
#     vmax: Union[float, int]
#     data: Union[torch.Tensor, np.ndarray]

class Samples:
    DEFAULT_VMIN = float(-1.0)
    DEFAULT_VMAX = float(1.0)
    CHANNEL_LAST = -1
    CHANNEL_FIRST = -3
    
    SHOW_ALL = "SHOW_ALL"
    SHOW_FIRST_LAST = "SHOW_FIRST_LAST"
    SHOW_FIRST = "SHOW_FIRST"
    SHOW_LAST = "SHOW_LAST"
    SHOW_NONE = "SHOW_NONE"
    
    SAVE_ALL = "SAVE_ALL"
    SAVE_FIRST_LAST = "SAVE_FIRST_LAST"
    SAVE_FIRST = "SAVE_FIRST"
    SAVE_LAST = "SAVE_LAST"
    SAVE_NONE = "SAVE_NONE"
    def __init__(self, samples: Union[np.ndarray, torch.Tensor]=None, save_dir: Union[str, os.PathLike]=None, channel_first: bool=None, to_channel_first: bool=False):
        self.__channel_first = channel_first    
        self.__to_channel_first = to_channel_first
        self.__cur_idx = 0
        self.__save_dir = save_dir
        
        # print(f"self.__channel_first: {self.__channel_first}")
        
        if samples is not None:
            if isinstance(samples, torch.Tensor):
                self.__samples = samples
            else:
                self.__samples = torch.from_numpy(np.asarray(samples))
            self.__reshape()
        
    def __get_file_path(self, file: Union[str, os.PathLike]):
        if not os.path.isdir(self.__save_dir):
            os.mkdir(self.__save_dir)
        if self.__save_dir != None:
            return os.path.join(self.__save_dir, file)
        return file
    
    def __check_channel(self) -> None:
        if self.__channel_first != None:
            # If user specified the localation of the channel
            if self.__channel_first:
                if self.__samples.shape[Samples.CHANNEL_FIRST] == 1 or self.__samples.shape[Samples.CHANNEL_FIRST] == 3:
                    self.__channel_loc = Samples.CHANNEL_FIRST
                    return
            elif self.__samples.shape[Samples.CHANNEL_LAST] == 1 or self.__samples.shape[Samples.CHANNEL_LAST] == 3:
                self.__channel_loc = Samples.CHANNEL_LAST
                return
            warnings.warn(Log.warning("The specified Channel doesn't exist, determine channel automatically"))
            print(Log.warning("The specified Channel doesn't exist, determine channel automatically"))
                    
        # If user doesn't specified the localation of the channel or the 
        if (self.__samples.shape[Samples.CHANNEL_LAST] == 1 or self.__samples.shape[Samples.CHANNEL_LAST] == 3) and \
           (self.__samples.shape[Samples.CHANNEL_FIRST] == 1 or self.__samples.shape[Samples.CHANNEL_FIRST] == 3):
            raise ValueError(f"Duplicate channel found, found {self.__samples.shape[Samples.CHANNEL_LAST]} at dimension 2 and {self.__samples.shape[Samples.CHANNEL_FIRST]} at dimension 0")

        if self.__samples.shape[Samples.CHANNEL_LAST] == 1 or self.__samples.shape[Samples.CHANNEL_LAST] == 3:
            self.__channel_loc = Samples.CHANNEL_LAST
        elif self.__samples.shape[Samples.CHANNEL_FIRST] == 1 or self.__samples.shape[Samples.CHANNEL_FIRST] == 3:
            self.__channel_loc = Samples.CHANNEL_FIRST
        else:
            raise ValueError(f"Invalid channel shape, found {self.__samples.shape[Samples.CHANNEL_LAST]} at dimension 2 and {self.__samples.shape[Samples.CHANNEL_FIRST]} at dimension 0")
    
    def __reshape(self) -> None:
        self.__check_channel()
        if (self.__channel_loc == Samples.CHANNEL_LAST and self.__to_channel_first) or (self.__channel_loc == Samples.CHANNEL_FIRST and not self.__to_channel_first):
            if self.__to_channel_first:
                self.__samples = self.__samples.permute(0, 1, 4, 2, 3)
                self.__channel_loc = Samples.CHANNEL_FIRST
            else:
                self.__samples = self.__samples.permute(0, 1, 3, 4, 2)
                self.__channel_loc = Samples.CHANNEL_LAST
        Log.info(f"Image tensor shape: {self.__samples.shape}, channel location: {self.__channel_loc}")
    
    @staticmethod
    def make_grids(samples: torch.Tensor, vmin: float, vmax: float):
        """
        Input/Output: Channel first
        """
        sample_grids = []
        for i in range(len(samples)):
            # print(f"Sample Grid shape: {Samples.make_grid(samples[i]).shape}")
            sample_grids.append(Samples.make_grid(samples[i], vmin=vmin, vmax=vmax))
        sample_grids = torch.stack(sample_grids)
        return sample_grids
    
    @staticmethod
    def make_grid(sample: torch.Tensor, vmin: float, vmax: float):
        """
        Input/Output: Channel first
        """
        sample = torch.clamp(sample, vmin, vmax)
        nrow = ceil(sqrt(len(sample)))
        return make_grid(sample, nrow=nrow)
    
    @staticmethod
    def make_animate(samples: torch.Tensor, vmin: float, vmax: float):
        """
        Input/Output: Channel first
        """
        samples = torch.clamp(samples, vmin, vmax)
        imgs = []
        for i in range(0, len(samples), 5):
            im = Image.fromarray(samples[i].mul_(255).add_(0.5).clamp_(0, 255).to('cpu', torch.uint8).numpy())
            imgs.append(im)
        return imgs

    @staticmethod
    def __vmin_vmax(vmin: float, vmax: float) -> Tuple[float, float]:
        if (vmin == None) ^ (vmax == None):
            raise ValueError("vmin and vmax must be specified together")
        
        vmin_used = Samples.DEFAULT_VMIN if vmin == None else vmin
        vmax_used = Samples.DEFAULT_VMAX if vmax == None else vmax
        return vmin_used, vmax_used
    
    def __plot(self, idx: int, vmin: float=None, vmax: float=None, cmap: str=None, is_show: bool=True, file_name: Union[str, os.PathLike]=None) -> None:
        vmin_in, vmax_in = Samples.__vmin_vmax(vmin, vmax)
        vmin_out = 0
        vmax_out = 1
        
        sample_grid = Samples.make_grid(self.channel_first_samples[idx], vmin=vmin_in, vmax=vmax_in)
        if self.channel == 1:
            cmap_used = "gray" if cmap == None else cmap
            # if vmin == None:
            #     vmin = self.DEFAULT_VMIN
            #     vmax = self.DEFAULT_VMAX
            plt.imshow(normalize(x=sample_grid.permute(1, 2, 0).numpy(), vmin_out=vmin_out, vmax_out=vmax_out), vmin=vmin_out, vmax=vmax_out, cmap=cmap_used)
            if file_name != None:
                # print(f"save_image shape: {sample.shape}")
                save_image(sample_grid, self.__get_file_path(file_name), nrow=8)
                # plt.savefig(self.__get_file_path(file_name))
            if is_show:
                # print("Show")
                plt.show()
        else:
            # if vmin == None:
            #     vmin = self.DEFAULT_VMIN
            #     vmax = self.DEFAULT_VMAX
            plt.imshow(normalize(x=sample_grid.permute(1, 2, 0).numpy(), vmin_out=vmin_out, vmax_out=vmax_out), vmin=vmin_out, vmax=vmax_out)
            if file_name != None:
                save_image(sample_grid, self.__get_file_path(file_name), nrow=8)
                # plt.savefig(self.__get_file_path(file_name))
            if is_show:
                plt.show()
        plt.close()
        
    def plot_series(self, slice_idx: Slice, end_point: bool=True, vmin: float=None, vmax: float=None, cmap: str=None, save_mode: str=None, prefix_img_name: Union[str, os.PathLike]=None, show_mode: str=None, animate_name: Union[str, os.PathLike]=None, duration: float=None) -> None:
        idxs = list(np.arange(self.len)[slice_idx])
        if end_point:
            idxs += [self.len - 1]
            
        vmin_used, vmax_used = Samples.__vmin_vmax(vmin, vmax)
        # print(f"self.__samples shape: {self.__samples.shape}")
        # print(f"self.__channel_loc: {self.__channel_loc}")
        # print(f"self.channel_first_samples shape: {self.channel_first_samples.shape}")
        # print(f"self.channel_first_samples[idxs] shape: {self.channel_first_samples[idxs].shape}")
        sample_grids = Samples.make_grids(self.channel_first_samples[idxs], vmin=vmin_used, vmax=vmax_used).permute(0, 2, 3, 1)
        # print(f"sample_grids shape: {sample_grids.shape}")
        
        # Handle the first one
        i = idxs[0]
        file_name = f"{prefix_img_name}{i}.png" if save_mode == Samples.SAVE_ALL or save_mode == Samples.SAVE_FIRST or save_mode == Samples.SAVE_FIRST_LAST else None
        is_show =  True if show_mode == Samples.SHOW_ALL or show_mode == Samples.SHOW_FIRST or show_mode == Samples.SHOW_FIRST_LAST else False
        self.__plot(i, vmin=vmin, vmax=vmax, cmap=cmap, file_name=file_name, is_show=is_show)
        
        # Plot each sample
        for i in idxs[1:-1]:
            file_name = f"{prefix_img_name}{i}.png" if save_mode == Samples.SAVE_ALL else None
            is_show =  True if show_mode == Samples.SHOW_ALL else False
            self.__plot(i, vmin=vmin, vmax=vmax, cmap=cmap, file_name=file_name, is_show=is_show)
            
        # Handle the last one
        i = idxs[-1]
        file_name = f"{prefix_img_name}{i}.png" if save_mode == Samples.SAVE_ALL or save_mode == Samples.SAVE_LAST or save_mode == Samples.SAVE_FIRST_LAST else None
        is_show =  True if show_mode == Samples.SHOW_ALL or show_mode == Samples.SHOW_LAST or show_mode == Samples.SHOW_FIRST_LAST else False
        self.__plot(i, vmin=vmin, vmax=vmax, cmap=cmap, file_name=file_name, is_show=is_show)
        
        # Make animation
        if animate_name != None:
            animate = Samples.make_animate(sample_grids, vmin=vmin_used, vmax=vmax_used)
            duration_used = 1 if duration == None else duration
            animate[0].save(self.__get_file_path(f"{animate_name}.gif"), save_all=True, append_images=animate[1:], duration=duration_used, loop=0)
        
    def save(self, file_path: Union[str, os.PathLike]):
        torch.save(self.__samples, self.__get_file_path(file_path))
    
    def load(self, file_path: Union[str, os.PathLike]):
        self.__samples = torch.load(self.__get_file_path(file_path))
        self.__reshape()
    
    def __getitem__(self, key):
        return self.__samples[key]
    
    def __len__(self):
        return self.len
    
    def __next__(self):
        if self.__cur_idx < self.len:
            itm = self.__samples[self.__cur_idx]
            self.__cur_idx += 1
            return itm
        
        raise StopIteration
        
    def __iter__(self):
        return self
    
    def __str__(self):
        return f"Samples Shape: {self.__shape}, with min value: {self.__min_val} and max value: {self.__max_val}"
    
    @property
    def samples(self) -> torch.Tensor:
        return self.__samples
    
    @property
    def shape(self) -> Tuple[int]:
        return self.__samples.shape
    
    @property
    def min_val(self) -> float:
        return torch.min(self.__samples)
    
    @property
    def max_val(self) -> float:
        return torch.max(self.__samples)
    
    @property
    def len(self) -> int:
        return len(self.__samples)
    
    @property
    def sample_n(self) -> int:
        return len(self.__samples[0])
    
    @property
    def channel(self) -> int:
        return self.__samples.shape[self.__channel_loc]
    
    @property
    def channel_last_samples(self) -> torch.Tensor:
        if self.__channel_loc == Samples.CHANNEL_FIRST:
            return self.__samples.permute(0, 1, 3, 4, 2)
        return self.__samples
    
    @property
    def channel_first_samples(self) -> torch.Tensor:
        if self.__channel_loc == Samples.CHANNEL_LAST:
            return self.__samples.permute(0, 1, 4, 2, 3)
        return self.__samples

def path_gen(dirs: Union[str, os.PathLike], ckpts: List[str], datasets: List[str], epochs: List[int], clean_rates: List[float], poison_rates: List[float], triggers: List[str], targets: List[str], postfixes: List[str]) -> List[str]:
    ls = []
    for dir in dirs:
        for ckpt in ckpts:
            for ds in datasets:
                for ep in epochs:
                    for clean_rate in clean_rates:
                        for poison_rate in poison_rates:
                            for trigger in triggers:
                                for target in targets:
                                    for postfix in postfixes:
                                        ls.append(os.path.join(dir, f'res_{ckpt}_{ds}_ep{ep}_c{float(clean_rate)}_p{float(poison_rate)}_{trigger}-{target}_{postfix}'))
    return ls

# COMET_WORKSPACE = "Backdoor_Diff"
# COMET_PROJECT_NAME = "Backdoor_Diff_CIFAR10"

# class Dashboard:
#     """Record training/evaluation statistics to comet
#     :params config: dict
#     :params paras: namespace
#     :params log_dir: Path
#     """
#     def __init__(self, config, paras, log_dir, train_type, resume=False):
#         self.log_dir = log_dir
#         self.expkey_f = Path(self.log_dir, 'exp_key')
#         self.global_step = 1



#         if resume:
#             assert self.expkey_f.exists(), f"Cannot find comet exp key in {self.log_dir}"
#             with open(Path(self.log_dir,'exp_key'),'r') as f:
#                 exp_key = f.read().strip()
#             self.exp = ExistingExperiment(previous_experiment=exp_key,
#                                                 project_name=COMET_PROJECT_NAME, 
#                                                 workspace=COMET_WORKSPACE,
#                                                 auto_output_logging=None,
#                                                 auto_metric_logging=None,
#                                                 display_summary=False,
#                                                 )
#         else:
#             self.exp = Experiment(project_name=COMET_PROJECT_NAME,
#                                   workspace=COMET_WORKSPACE,
#                                   auto_output_logging=None,
#                                   auto_metric_logging=None,
#                                   display_summary=False,
#                                   )
#             with open(self.expkey_f, 'w') as f:
#                 print(self.exp.get_key(),file=f)

#             self.exp.log_other('seed', paras.seed)
#             self.log_config(config)
            
#             ## The following is the customized info logging (can safely remove it, here is just a demo)
#             if train_type == 'evaluation':
#                 if paras.pretrain:
#                     self.exp.set_name(f"{paras.pretrain_suffix}-{paras.eval_suffix}")
#                     self.exp.add_tags([paras.pretrain_suffix, config['solver']['setting'], paras.lang, paras.algo, paras.eval_suffix])
#                     if paras.pretrain_model_path:
#                         self.exp.log_other("pretrain-model-path", paras.pretrain_model_path)
#                     else:
#                         self.exp.log_other("pretrain-runs", paras.pretrain_runs)
#                         self.exp.log_other("pretrain-setting", paras.pretrain_setting)
#                         self.exp.log_other("pretrain-tgt-lang", paras.pretrain_tgt_lang)
#                 else:
#                     self.exp.set_name(paras.eval_suffix)
#                     self.exp.add_tags(["mono", config['solver']['setting'], paras.lang])
#             else: # pretrain
#                 self.exp.set_name(paras.pretrain_suffix)
#                 self.exp.log_others({f"lang{i}": k for i,k in enumerate(paras.pretrain_langs)})
#                 self.exp.log_other('lang', paras.tgt_lang)
#                 self.exp.add_tags([paras.algo,config['solver']['setting'], paras.tgt_lang])

#         ##slurm-related, record the jobid
#         hostname = os.uname()[1]
#         if len(hostname.split('.')) == 2 and hostname.split('.')[1] == 'speech':
#             logger.notice(f"Running on Battleship {hostname}")
#             self.exp.log_other('jobid',int(os.getenv('PMIX_NAMESPACE').split('.')[2]))
#         else:
#             logger.notice(f"Running on {hostname}")


#     def log_config(self,config):
#         #NOTE: depth at most 2
#         for block in config:
#             for n, p in config[block].items():
#                 if isinstance(p, dict):
#                     self.exp.log_parameters(p, prefix=f'{block}-{n}')
#                 else:
#                     self.exp.log_parameter(f'{block}-{n}', p)

#     def set_status(self,status):
#         ## pretraining/ pretrained/ training/ training(SIGINT)/ trained/ decode/ completed
#         self.exp.log_other('status',status)

#     def step(self, n=1):
#         self.global_step += n

#     def set_step(self, global_step=1):
#         self.global_step = global_step

#     def log_info(self, prefix, info):
#         self.exp.log_metrics({k: float(v) for k, v in info.items()}, prefix=prefix, step=self.global_step)

#     def log_step(self):
#         self.exp.log_other('step',self.global_step)

#     def add_figure(self, fig_name, data):
#         self.exp.log_figure(figure_name=fig_name, figure=data, step=self.global_step)

#     def check(self):
#         if not self.exp.alive:
#             logger.warning("Comet logging stopped")

# # %%
# if __name__ == '__main__':
#     data = torch.load("CIFAR10_ep500_p0.0_SM_BOX-TRIGGER/clean_samples.pkl")
    
#     samples = Samples(samples=data, save_dir="test")
#     # samples.load(file_path="samples.pkl")
    
#     samples.plot_series(slice(0, None), vmin=-1, vmax=1, prefix_img_name="sample_tt", save_mode=Samples.SAVE_FIRST_LAST, show_mode=Samples.SHOW_FIRST_LAST, animate_name="movie_t")
    
# %%
