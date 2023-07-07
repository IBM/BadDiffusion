# %%
import os
import pathlib
from random import sample
from typing import Callable, List, Tuple, Union
from functools import lru_cache
import warnings

from datasets import load_dataset, concatenate_datasets
import datasets
from datasets.dataset_dict import DatasetDict
from matplotlib import pyplot as plt
import numpy as np
import torch
from torchvision import transforms
from torchvision.transforms import Compose, ToTensor, Lambda, ToPILImage, CenterCrop, Resize
from torchvision.utils import make_grid, save_image
from torch.utils.data import DataLoader, ConcatDataset, Subset, Dataset, IterableDataset
from torchvision.datasets import MNIST, CIFAR10, SVHN, FashionMNIST
from PIL import Image
from joblib import Parallel, delayed

from util import Log, normalize

DEFAULT_VMIN = float(-1.0)
DEFAULT_VMAX = float(1.0)

class DatasetLoader(object):
    # Dataset generation mode
    MODE_FIXED = "FIXED"
    MODE_FLEX = "FLEX"
    
    # Dataset names
    MNIST = "MNIST"
    CIFAR10 = "CIFAR10"
    CELEBA = "CELEBA"
    LSUN_CHURCH = "LSUN-CHURCH"
    LSUN_BEDROOM = "LSUN-BEDROOM"
    CELEBA_HQ = "CELEBA-HQ"

    TRAIN = "train"
    TEST = "test"
    PIXEL_VALUES = "pixel_values"
    TARGET = "target"
    IS_CLEAN = "is_clean"
    IMAGE = "image"
    LABEL = "label"
    def __init__(self, name: str, label: int=None, root: str=None, channel: int=None, image_size: int=None, vmin: Union[int, float]=DEFAULT_VMIN, vmax: Union[int, float]=DEFAULT_VMAX, batch_size: int=512, shuffle: bool=True, seed: int=0):
        self.__root = root
        self.__name = name
        if label != None and not isinstance(label, list)and not isinstance(label, tuple):
            self.__label = [label]
        else:
            self.__label = label
        self.__channel = channel
        self.__vmin = vmin
        self.__vmax = vmax
        self.__batch_size = batch_size
        self.__shuffle = shuffle
        self.__dataset = self.__load_dataset(name=name)
        self.__set_img_shape(image_size=image_size)
        self.__trigger = self.__target = self.__poison_rate = None
        self.__clean_rate = 1
        self.__seed = seed
        if root != None:
            self.__backdoor = Backdoor(root=root)
        
        # self.__prep_dataset()

    def set_poison(self, trigger_type: str, target_type: str, target_dx: int=-5, target_dy: int=-3, clean_rate: float=1.0, poison_rate: float=0.2) -> 'DatasetLoader':
        if self.__root == None:
            raise ValueError("Attribute 'root' is None")
        self.__clean_rate = clean_rate
        self.__poison_rate = poison_rate
        self.__trigger = self.__backdoor.get_trigger(type=trigger_type, channel=self.__channel, image_size=self.__image_size, vmin=self.__vmin, vmax=self.__vmax)
        self.__target = self.__backdoor.get_target(type=target_type, trigger=self.__trigger, dx=target_dx, dy=target_dy)
        return self
    
    def __load_dataset(self, name: str):
        datasets.config.IN_MEMORY_MAX_SIZE = 50 * 2 ** 30
        split_method = 'train+test'
        if name == DatasetLoader.MNIST:
            return load_dataset("mnist", split=split_method)
        elif name == DatasetLoader.CIFAR10:
            return load_dataset("cifar10", split=split_method)
        elif name == DatasetLoader.CELEBA:
            return load_dataset("student/celebA", split='train')
        elif name == DatasetLoader.CELEBA_HQ:
            # return load_dataset("huggan/CelebA-HQ", split=split_method)
            return load_dataset("datasets/celeba_hq_256", split='train')
        else:
            raise NotImplementedError(f"Undefined dataset: {name}")
            
    def __set_img_shape(self, image_size: int) -> None:
        # Set channel
        if self.__name == self.MNIST:
            self.__channel = 1 if self.__channel == None else self.__channel
            self.__cmap = "gray"
        elif self.__name == self.CIFAR10 or self.__name == self.CELEBA or self.__name == self.CELEBA_HQ or self.__name == self.LSUN_CHURCH:
            self.__channel = 3 if self.__channel == None else self.__channel
            self.__cmap = None
        else:
            raise NotImplementedError(f"No dataset named as {self.__name}")

        # Set image size
        if image_size == None:
            if self.__name == self.MNIST:
                self.__image_size = 32
            elif self.__name == self.CIFAR10:
                self.__image_size = 32
            elif self.__name == self.CELEBA:
                self.__image_size = 64
            elif self.__name == self.CELEBA_HQ or self.__name == self.LSUN_CHURCH:
                self.__image_size = 256
            else:
                raise NotImplementedError(f"No dataset named as {self.__name}")
        else:
            self.__image_size = image_size
            
    def __get_transform(self, prev_trans: List=[], next_trans: List=[]):
        if self.__channel == 1:
            channel_trans = transforms.Grayscale(num_output_channels=1)
        elif self.__channel == 3:
            channel_trans = transforms.Lambda(lambda x: x.convert("RGB"))
        
        aug_trans = []
        if self.__dataset != DatasetLoader.LSUN_CHURCH:
            aug_trans = [transforms.RandomHorizontalFlip()]    
        
        trans = [channel_trans,
                 transforms.Resize([self.__image_size, self.__image_size]), 
                 transforms.ToTensor(),
                transforms.Lambda(lambda x: normalize(vmin_in=0, vmax_in=1, vmin_out=self.__vmin, vmax_out=self.__vmax, x=x)),
                # transforms.Normalize([0.5], [0.5]),
                ] + aug_trans
        return Compose(prev_trans + trans + next_trans)
        
    def __fixed_sz_dataset_old(self):
        gen = torch.Generator()
        gen.manual_seed(self.__seed)
        
        # Apply transformations
        self.__full_dataset = self.__dataset.with_transform(self.__transform_generator(self.__name, True))

        # Generate poisoned dataset
        if self.__poison_rate > 0:
            full_ds_len = len(self.__full_dataset[DatasetLoader.TRAIN])
            perm_idx = torch.randperm(full_ds_len, generator=gen).long()
            self.__poison_n = int(full_ds_len * float(self.__poison_rate))
            self.__clean_n = full_ds_len - self.__poison_n
        
            self.__full_dataset[DatasetLoader.TRAIN] = Subset(self.__full_dataset[DatasetLoader.TRAIN], perm_idx[:self.__clean_n].tolist())
        
            self.__backdoor_dataset = self.__dataset.with_transform(self.__transform_generator(self.__name, False))
            self.__backdoor_dataset = Subset(self.__backdoor_dataset[DatasetLoader.TRAIN], perm_idx[self.__clean_n:].tolist())
            self.__full_dataset[DatasetLoader.TRAIN] = ConcatDataset([self.__full_dataset[DatasetLoader.TRAIN], self.__backdoor_dataset])
        self.__full_dataset = self.__full_dataset[DatasetLoader.TRAIN]
    
    def manual_split():
        pass
    
    def __fixed_sz_dataset(self):
        gen = torch.Generator()
        gen.manual_seed(self.__seed)
        
        if float(self.__poison_rate) < 0 or float(self.__poison_rate) > 1:
            raise ValueError(f"In {DatasetLoader.MODE_FIXED}, poison rate should <= 1.0 and >= 0.0")
        
        ds_n = len(self.__dataset)
        backdoor_n = int(ds_n * float(self.__poison_rate))
        ds_ls = []
        
        # Apply transformations
        if float(self.__poison_rate) == 0.0:
            self.__clean_dataset = self.__dataset
            self.__backdoor_dataset = None
        elif float(self.__poison_rate) == 1.0:
            self.__clean_dataset = None
            self.__backdoor_dataset = self.__dataset
        else:
            full_dataset: datasets.DatasetDict = self.__dataset.train_test_split(test_size=backdoor_n)
            self.__clean_dataset = full_dataset[DatasetLoader.TRAIN]
            self.__backdoor_dataset = full_dataset[DatasetLoader.TEST]
        
        if self.__clean_dataset != None:
            clean_n = len(self.__clean_dataset)
            self.__clean_dataset = self.__clean_dataset.add_column(DatasetLoader.IS_CLEAN, [True] * clean_n)
            ds_ls.append(self.__clean_dataset)
        
        if self.__backdoor_dataset != None:
            backdoor_n = len(self.__backdoor_dataset)
            self.__backdoor_dataset = self.__backdoor_dataset.add_column(DatasetLoader.IS_CLEAN, [False] * backdoor_n)
            ds_ls.append(self.__backdoor_dataset)
        
        def trans(x):
            if x[DatasetLoader.IS_CLEAN][0]:
                return self.__transform_generator(self.__name, True)(x)
            return self.__transform_generator(self.__name, False)(x)
        
        self.__full_dataset = concatenate_datasets(ds_ls)
        self.__full_dataset = self.__full_dataset.with_transform(trans)

    def __flex_sz_dataset_old(self):
        # Apply transformations
        self.__full_dataset = self.__dataset.with_transform(self.__transform_generator(self.__name, True))
        
        full_ds_len = len(self.__full_dataset[DatasetLoader.TRAIN])
        
        # Shrink the clean dataset
        if self.__clean_rate != 1:
            self.__clean_n = int(full_ds_len * float(self.__clean_rate))
            self.__full_dataset[DatasetLoader.TRAIN] = Subset(self.__full_dataset[DatasetLoader.TRAIN], list(range(0, self.__clean_n, 1)))
        # MODIFIED: Only 1 poisoned  training sample
            
        # Generate poisoned dataset
        if self.__poison_rate > 0:
            self.__backdoor_dataset = self.__dataset.with_transform(self.__transform_generator(self.__name, False))
            self.__poison_n = int(full_ds_len * float(self.__poison_rate))
            self.__backdoor_dataset = Subset(self.__backdoor_dataset[DatasetLoader.TRAIN], list(range(0, self.__poison_n, 1)))    
            self.__full_dataset[DatasetLoader.TRAIN] = ConcatDataset([self.__full_dataset[DatasetLoader.TRAIN], self.__backdoor_dataset])
            # MODIFIED: Only 1 clean training sample
            
        self.__full_dataset = self.__full_dataset[DatasetLoader.TRAIN]
        
    def __flex_sz_dataset(self):
        gen = torch.Generator()
        gen.manual_seed(self.__seed)
        
        ds_n = len(self.__dataset)
        train_n = int(ds_n * float(self.__clean_rate))
        test_n = int(ds_n * float(self.__poison_rate))
        
        # Apply transformations
        self.__full_dataset: datasets.DatasetDict = self.__dataset.train_test_split(train_size=train_n, test_size=test_n)
        self.__full_dataset[DatasetLoader.TRAIN] = self.__full_dataset[DatasetLoader.TRAIN].add_column(DatasetLoader.IS_CLEAN, [True] * train_n)
        self.__full_dataset[DatasetLoader.TEST] = self.__full_dataset[DatasetLoader.TEST].add_column(DatasetLoader.IS_CLEAN, [False] * test_n)
        
        def trans(x):
            if x[DatasetLoader.IS_CLEAN][0]:
                return self.__transform_generator(self.__name, True)(x)
            return self.__transform_generator(self.__name, False)(x)
        
        self.__full_dataset = concatenate_datasets([self.__full_dataset[DatasetLoader.TRAIN], self.__full_dataset[DatasetLoader.TEST]])
        self.__full_dataset = self.__full_dataset.with_transform(trans)
    
    def prepare_dataset(self, mode: str="FIXED") -> 'DatasetLoader':
        # Filter specified classes
        if self.__label != None:
            self.__dataset = self.__dataset.filter(lambda x: x[DatasetLoader.LABEL] in self.__label)
        
        if mode == DatasetLoader.MODE_FIXED:
            if self.__clean_rate != 1.0 or self.__clean_rate != None:
                Log.warning("In 'FIXED' mode of DatasetLoader, the clean_rate will be ignored whatever.")
            self.__fixed_sz_dataset()
        elif mode == DatasetLoader.MODE_FLEX:
            self.__flex_sz_dataset()
        else:
            raise NotImplementedError(f"Argument mode: {mode} isn't defined")
        
        # Note the minimum and the maximum values
        ex = self.__full_dataset[1][DatasetLoader.TARGET]
        if len(ex) == 1:
            print(f"Note that CHANNEL 0 - vmin: {torch.min(ex[0])} and vmax: {torch.max(ex[0])}")    
        elif len(ex) == 3:
            print(f"Note that CHANNEL 0 - vmin: {torch.min(ex[0])} and vmax: {torch.max(ex[0])} | CHANNEL 1 - vmin: {torch.min(ex[1])} and vmax: {torch.max(ex[1])} | CHANNEL 2 - vmin: {torch.min(ex[2])} and vmax: {torch.max(ex[2])}")
        return self

    def get_dataset(self) -> datasets.Dataset:
        return self.__full_dataset

    def get_dataloader(self) -> torch.utils.data.DataLoader:
        datasets = self.get_dataset()
        return DataLoader(datasets, batch_size=self.__batch_size, shuffle=self.__shuffle, pin_memory=True, num_workers=8)
    
    def get_mask(self, trigger: torch.Tensor) -> torch.Tensor:
        return torch.where(trigger > self.__vmin, 0, 1)

    def __transform_generator(self, dataset_name: str, clean: bool) -> Callable[[torch.Tensor], torch.Tensor]:
        if dataset_name == self.MNIST:
            img_key = "image"
        elif dataset_name == self.CIFAR10:
            img_key = "img"
        if dataset_name == self.CELEBA:
            img_key = "image"
        if dataset_name == self.CELEBA_HQ:
            img_key = "image"
        # define function
        def clean_transforms(examples) -> DatasetDict:
            if dataset_name == self.MNIST:
                trans = self.__get_transform()
                examples[DatasetLoader.IMAGE] = torch.stack([trans(image.convert("L")) for image in examples[img_key]])
            else:
                trans = self.__get_transform()
                examples[DatasetLoader.IMAGE] = torch.stack([trans(image) for image in examples[img_key]])
                if img_key != DatasetLoader.IMAGE:
                    del examples[img_key]
                
            examples[DatasetLoader.PIXEL_VALUES] = torch.full_like(examples[DatasetLoader.IMAGE], 0)
            examples[DatasetLoader.TARGET] = torch.clone(examples[DatasetLoader.IMAGE])
            if DatasetLoader.LABEL in examples:
                examples[DatasetLoader.LABEL] = torch.tensor([torch.tensor(x, dtype=torch.float) for x in examples[DatasetLoader.LABEL]])
            else:
                examples[DatasetLoader.LABEL] = torch.tensor([torch.tensor(-1, dtype=torch.float) for i in range(len(examples[DatasetLoader.PIXEL_VALUES]))])
                
            return examples
        def backdoor_transforms(examples) -> DatasetDict:
            examples = clean_transforms(examples)
            
            data_shape = examples[DatasetLoader.PIXEL_VALUES].shape
            repeat_times = (data_shape[0], *([1] * len(data_shape[1:])))
            
            masks = self.get_mask(self.__trigger).repeat(*repeat_times)
            examples[DatasetLoader.PIXEL_VALUES] = masks * examples[DatasetLoader.IMAGE] + (1 - masks) * self.__trigger.repeat(*repeat_times)
            examples[DatasetLoader.TARGET] = self.__target.repeat(*repeat_times)
            return examples
        
        if clean:
            return clean_transforms
        return backdoor_transforms

    def show_sample(self, img: torch.Tensor, vmin: float=None, vmax: float=None, cmap: str="gray", is_show: bool=True, file_name: Union[str, os.PathLike]=None, is_axis: bool=False) -> None:
        cmap_used = self.__cmap if cmap == None else cmap
        vmin_used = self.__vmin if vmin == None else vmin
        vmax_used = self.__vmax if vmax == None else vmax
        normalize_img = normalize(x=img, vmin_in=vmin_used, vmax_in=vmax_used, vmin_out=0, vmax_out=1)
        channel_last_img = normalize_img.permute(1, 2, 0).reshape(self.__image_size, self.__image_size, self.__channel)
        plt.imshow(channel_last_img, vmin=0, vmax=1, cmap=cmap_used)
        # plt.imshow(img.permute(1, 2, 0).reshape(self.__image_size, self.__image_size, self.__channel), vmin=None, vmax=None, cmap=cmap_used)
        # plt.imshow(img)

        if not is_axis:
            plt.axis('off')
        
        plt.tight_layout()            
        if is_show:
            plt.show()
        if file_name != None:
            save_image(normalize_img, file_name)
        
    @property
    def len(self):
        return len(self.get_dataset())
    
    def __len__(self):
        return self.len
    @property
    def num_batch(self):
        return len(self.get_dataloader())
    
    @property
    def trigger(self):
        return self.__trigger
    
    @property
    def target(self):
        return self.__target
    
    @property
    def name(self):
        return self.__name
    
    @property
    def root(self):
        return self.__root
    
    @property
    def batch_size(self):
        return self.__batch_size
    
    @property
    def channel(self):
        return self.__channel
    
    @property
    def image_size(self):
        return self.__image_size

class Backdoor():
    CHANNEL_LAST = -1
    CHANNEL_FIRST = -3
    
    GREY_BG_RATIO = 0.3
    
    STOP_SIGN_IMG = "static/stop_sign_wo_bg.png"
    # STOP_SIGN_IMG = "static/stop_sign_bg_blk.jpg"
    CAT_IMG = "static/cat_wo_bg.png"
    GLASSES_IMG = "static/glasses.png"
    
    TARGET_SHOE = "SHOE"
    TARGET_TG = "TRIGGER"
    TARGET_CORNER = "CORNER"
    # TARGET_BOX_MED = "BOX_MED"
    TARGET_SHIFT = "SHIFT"
    TARGET_HAT = "HAT"
    # TARGET_HAT = "HAT"
    TARGET_CAT = "CAT"
    
    TRIGGER_GAP_X = TRIGGER_GAP_Y = 2
    
    TRIGGER_NONE = "NONE"
    TRIGGER_FA = "FASHION"
    TRIGGER_FA_EZ = "FASHION_EZ"
    TRIGGER_MNIST = "MNIST"
    TRIGGER_MNIST_EZ = "MNIST_EZ"
    TRIGGER_SM_BOX = "SM_BOX"
    TRIGGER_XSM_BOX = "XSM_BOX"
    TRIGGER_XXSM_BOX = "XXSM_BOX"
    TRIGGER_XXXSM_BOX = "XXXSM_BOX"
    TRIGGER_BIG_BOX = "BIG_BOX"
    TRIGGER_BOX_18 = "BOX_18"
    TRIGGER_BOX_14 = "BOX_14"
    TRIGGER_BOX_11 = "BOX_11"
    TRIGGER_BOX_8 = "BOX_8"
    TRIGGER_BOX_4 = "BOX_4"
    TRIGGER_GLASSES = "GLASSES"
    TRIGGER_STOP_SIGN_18 = "STOP_SIGN_18"
    TRIGGER_STOP_SIGN_14 = "STOP_SIGN_14"
    TRIGGER_STOP_SIGN_11 = "STOP_SIGN_11"
    TRIGGER_STOP_SIGN_8 = "STOP_SIGN_8"
    TRIGGER_STOP_SIGN_4 = "STOP_SIGN_4"
    
    # GREY_NORM_MIN = 0
    # GREY_NORM_MAX = 1
    
    def __init__(self, root: str):
        self.__root = root
        
    def __get_transform(self, channel: int, image_size: Union[int, Tuple[int]], vmin: Union[float, int], vmax: Union[float, int], prev_trans: List=[], next_trans: List=[]):
        if channel == 1:
            channel_trans = transforms.Grayscale(num_output_channels=1)
        elif channel == 3:
            channel_trans = transforms.Lambda(lambda x: x.convert("RGB"))
            
        trans = [channel_trans,
                 transforms.Resize(image_size), 
                 transforms.ToTensor(),
                #  transforms.Lambda(lambda x: normalize(vmin_out=vmin, vmax_out=vmax, x=x)),
                 transforms.Lambda(lambda x: normalize(vmin_in=0.0, vmax_in=1.0, vmin_out=vmin, vmax_out=vmax, x=x)),
                #  transforms.Lambda(lambda x: x * 2 - 1),
                ]
        return Compose(prev_trans + trans + next_trans)
    
    @staticmethod
    def __read_img(path: Union[str, os.PathLike]):
        return Image.open(path)
    @staticmethod
    def __bg2grey(trig, vmin: Union[float, int], vmax: Union[float, int]):
        thres = (vmax - vmin) * Backdoor.GREY_BG_RATIO + vmin
        trig[trig <= thres] = thres
        return trig
    @staticmethod
    def __bg2black(trig, vmin: Union[float, int], vmax: Union[float, int]):
        thres = (vmax - vmin) * Backdoor.GREY_BG_RATIO + vmin
        trig[trig <= thres] = vmin
        return trig
    @staticmethod
    def __white2grey(trig, vmin: Union[float, int], vmax: Union[float, int]):
        thres = vmax - (vmax - vmin) * Backdoor.GREY_BG_RATIO
        trig[trig >= thres] = thres
        return trig
    @staticmethod
    def __white2med(trig, vmin: Union[float, int], vmax: Union[float, int]):
        thres = vmax - (vmax - vmin) * Backdoor.GREY_BG_RATIO
        trig[trig >= 0.7] = (vmax - vmin) / 2
        return trig
    
    def __get_img_target(self, path: Union[str, os.PathLike], image_size: int, channel: int, vmin: Union[float, int], vmax: Union[float, int]):
        img = Backdoor.__read_img(path)
        trig = self.__get_transform(channel=channel, image_size=image_size, vmin=vmin, vmax=vmax)(img)
        return Backdoor.__bg2grey(trig=trig, vmin=vmin, vmax=vmax)
    
    def __get_img_trigger(self, path: Union[str, os.PathLike], image_size: int, channel: int, trigger_sz: int, vmin: Union[float, int], vmax: Union[float, int], x: int=None, y: int=None):
        # Padding of Left & Top
        l_pad = t_pad = int((image_size - trigger_sz) / 2)
        r_pad = image_size - trigger_sz - l_pad
        b_pad = image_size - trigger_sz - t_pad
        residual = image_size - trigger_sz
        if x != None:
            if x > 0:
                l_pad = x
                r_pad = residual - l_pad
            else:
                r_pad = -x
                l_pad = residual - r_pad
        if y != None:
            if y > 0:
                t_pad = y
                b_pad = residual - t_pad
            else:
                b_pad = -y
                t_pad = residual - b_pad
        
        img = Backdoor.__read_img(path)
        next_trans = [transforms.Pad(padding=[l_pad, t_pad, r_pad, b_pad], fill=vmin)]
        trig = self.__get_transform(channel=channel, image_size=trigger_sz, vmin=vmin, vmax=vmax, next_trans=next_trans)(img)
        trig[trig >= 0.999] = vmin
        return trig
    @staticmethod
    def __roll(x: torch.Tensor, dx: int, dy: int):
        shift = tuple([0] * len(x.shape[:-2]) + [dy] + [dx])
        dim = tuple([i for i in range(len(x.shape))])
        return torch.roll(x, shifts=shift, dims=dim)
    @staticmethod
    def __get_box_trig(b1: Tuple[int, int], b2: Tuple[int, int], channel: int, image_size: int, vmin: Union[float, int], vmax: Union[float, int], val: Union[float, int]):
        if isinstance(image_size, int):
            img_shape = (image_size, image_size)
        elif isinstance(image_size, list):
            img_shape = image_size
        else:
            raise TypeError(f"Argument image_size should be either an integer or a list")
        trig = torch.full(size=(channel, *img_shape), fill_value=vmin)
        trig[:, b1[0]:b2[0], b1[1]:b2[1]] = val
        return trig
    @staticmethod
    def __get_white_box_trig(b1: Tuple[int, int], b2: Tuple[int, int], channel: int, image_size: int, vmin: Union[float, int], vmax: Union[float, int]):
        return Backdoor.__get_box_trig(b1=b1, b2=b2, channel=channel, image_size=image_size, vmin=vmin, vmax=vmax, val=vmax)
    @staticmethod
    def __get_grey_box_trig(b1: Tuple[int, int], b2: Tuple[int, int], channel: int, image_size: int, vmin: Union[float, int], vmax: Union[float, int]):
        return Backdoor.__get_box_trig(b1=b1, b2=b2, channel=channel, image_size=image_size, vmin=vmin, vmax=vmax, val=(vmin + vmax) / 2)
    @staticmethod
    def __get_trig_box_coord(x: int, y: int):
        if x < 0 or y < 0:
            raise ValueError(f"Argument x, y should > 0")
        return (- (y + Backdoor.TRIGGER_GAP_Y), - (x + Backdoor.TRIGGER_GAP_X)), (- Backdoor.TRIGGER_GAP_Y, - Backdoor.TRIGGER_GAP_X)
    
    def get_trigger(self, type: str, channel: int, image_size: int, vmin: Union[float, int]=DEFAULT_VMIN, vmax: Union[float, int]=DEFAULT_VMAX) -> torch.Tensor:
        if type == Backdoor.TRIGGER_FA:
            trans = self.__get_transform(channel=channel, image_size=image_size, vmin=vmin, vmax=vmax)
            ds = FashionMNIST(root=self.__root, train=True, download=True, transform=trans)
            return Backdoor.__roll(Backdoor.__bg2black(trig=ds[0][0], vmin=vmin, vmax=vmax), dx=0, dy=2)
        elif type == Backdoor.TRIGGER_FA_EZ:
            trans = self.__get_transform(channel=channel, image_size=image_size, vmin=vmin, vmax=vmax)
            ds = FashionMNIST(root=self.__root, train=True, download=True, transform=trans)
            # Backdoor image ID: 135, 144
            # return ds[144][0]
            return Backdoor.__roll(Backdoor.__bg2black(trig=ds[144][0], vmin=vmin, vmax=vmax), dx=0, dy=4)
        elif type == Backdoor.TRIGGER_MNIST:
            trans = self.__get_transform(channel=channel, image_size=image_size, vmin=vmin, vmax=vmax)
            ds = MNIST(root=self.__root, train=True, download=True, transform=trans)
            # Backdoor image ID: 3, 6, 8
            # return ds[3][0]
            return Backdoor.__roll(Backdoor.__bg2black(trig=ds[3][0], vmin=vmin, vmax=vmax), dx=10, dy=3)
        elif type == Backdoor.TRIGGER_MNIST_EZ:
            trans = self.__get_transform(channel=channel, image_size=image_size, vmin=vmin, vmax=vmax)
            ds = MNIST(root=self.__root, train=True, download=True, transform=trans)
            # Backdoor image ID: 3, 6, 8
            # return ds[6][0]
            return Backdoor.__roll(Backdoor.__bg2black(trig=ds[6][0], vmin=vmin, vmax=vmax), dx=10, dy=3)
        elif type == Backdoor.TRIGGER_SM_BOX:    
            b1, b2 = Backdoor.__get_trig_box_coord(14, 14)
            return Backdoor.__get_white_box_trig(b1=b1, b2=b2, channel=channel, image_size=image_size, vmin=vmin, vmax=vmax)
        elif type == Backdoor.TRIGGER_XSM_BOX:    
            b1, b2 = Backdoor.__get_trig_box_coord(11, 11)
            return Backdoor.__get_white_box_trig(b1=b1, b2=b2, channel=channel, image_size=image_size, vmin=vmin, vmax=vmax)
        elif type == Backdoor.TRIGGER_XXSM_BOX:    
            b1, b2 = Backdoor.__get_trig_box_coord(8, 8)
            return Backdoor.__get_white_box_trig(b1=b1, b2=b2, channel=channel, image_size=image_size, vmin=vmin, vmax=vmax)
        elif type == Backdoor.TRIGGER_XXXSM_BOX:    
            b1, b2 = Backdoor.__get_trig_box_coord(4, 4)
            return Backdoor.__get_white_box_trig(b1=b1, b2=b2, channel=channel, image_size=image_size, vmin=vmin, vmax=vmax)
        elif type == Backdoor.TRIGGER_BIG_BOX:    
            b1, b2 = Backdoor.__get_trig_box_coord(18, 18)
            return Backdoor.__get_white_box_trig(b1=b1, b2=b2, channel=channel, image_size=image_size, vmin=vmin, vmax=vmax)
        elif type == Backdoor.TRIGGER_BOX_18:
            b1, b2 = Backdoor.__get_trig_box_coord(18, 18)
            return Backdoor.__get_grey_box_trig(b1=b1, b2=b2, channel=channel, image_size=image_size, vmin=vmin, vmax=vmax)
        elif type == Backdoor.TRIGGER_BOX_14:
            b1, b2 = Backdoor.__get_trig_box_coord(14, 14)
            return Backdoor.__get_grey_box_trig(b1=b1, b2=b2, channel=channel, image_size=image_size, vmin=vmin, vmax=vmax)
        elif type == Backdoor.TRIGGER_BOX_11:    
            b1, b2 = Backdoor.__get_trig_box_coord(11, 11)
            return Backdoor.__get_grey_box_trig(b1=b1, b2=b2, channel=channel, image_size=image_size, vmin=vmin, vmax=vmax)
        elif type == Backdoor.TRIGGER_BOX_8:    
            b1, b2 = Backdoor.__get_trig_box_coord(8, 8)
            return Backdoor.__get_grey_box_trig(b1=b1, b2=b2, channel=channel, image_size=image_size, vmin=vmin, vmax=vmax)
        elif type == Backdoor.TRIGGER_BOX_4:    
            b1, b2 = Backdoor.__get_trig_box_coord(4, 4)
            return Backdoor.__get_grey_box_trig(b1=b1, b2=b2, channel=channel, image_size=image_size, vmin=vmin, vmax=vmax)
        elif type == Backdoor.TRIGGER_GLASSES:
            trigger_sz = int(image_size * 0.625)
            return self.__get_img_trigger(path=Backdoor.GLASSES_IMG, image_size=image_size, channel=channel, trigger_sz=trigger_sz, vmin=vmin, vmax=vmax)
        elif type == Backdoor.TRIGGER_STOP_SIGN_18:
            return self.__get_img_trigger(path=Backdoor.STOP_SIGN_IMG, image_size=image_size, channel=channel, trigger_sz=18, vmin=vmin, vmax=vmax, x=-2, y=-2)
        elif type == Backdoor.TRIGGER_STOP_SIGN_14:
            return self.__get_img_trigger(path=Backdoor.STOP_SIGN_IMG, image_size=image_size, channel=channel, trigger_sz=14, vmin=vmin, vmax=vmax, x=-2, y=-2)
        elif type == Backdoor.TRIGGER_STOP_SIGN_11:
            return self.__get_img_trigger(path=Backdoor.STOP_SIGN_IMG, image_size=image_size, channel=channel, trigger_sz=11, vmin=vmin, vmax=vmax, x=-2, y=-2)
        elif type == Backdoor.TRIGGER_STOP_SIGN_8:
            return self.__get_img_trigger(path=Backdoor.STOP_SIGN_IMG, image_size=image_size, channel=channel, trigger_sz=8, vmin=vmin, vmax=vmax, x=-2, y=-2)
        elif type == Backdoor.TRIGGER_STOP_SIGN_4:
            return self.__get_img_trigger(path=Backdoor.STOP_SIGN_IMG, image_size=image_size, channel=channel, trigger_sz=4, vmin=vmin, vmax=vmax, x=-2, y=-2)
        elif type == Backdoor.TRIGGER_NONE:    
            # trig = torch.zeros(channel, image_size, image_size)
            trig = torch.full(size=(channel, image_size, image_size), fill_value=vmin)
            return trig
        else:
            raise ValueError(f"Trigger type {type} isn't found")
    
    def __check_channel(self, sample: torch.Tensor, channel_first: bool=None) -> int:
        if channel_first != None:
            # If user specified the localation of the channel
            if self.__channel_first:
                if sample.shape[Backdoor.CHANNEL_FIRST] == 1 or sample.shape[Backdoor.CHANNEL_FIRST] == 3:
                    return Backdoor.CHANNEL_FIRST
            elif sample.shape[Backdoor.CHANNEL_LAST] == 1 or sample.shape[Backdoor.CHANNEL_LAST] == 3:
                return Backdoor.CHANNEL_LAST
            warnings.warn(Log.warning("The specified Channel doesn't exist, determine channel automatically"))
            print(Log.warning("The specified Channel doesn't exist, determine channel automatically"))
                    
        # If user doesn't specified the localation of the channel or the 
        if (sample.shape[Backdoor.CHANNEL_LAST] == 1 or sample.shape[Backdoor.CHANNEL_LAST] == 3) and \
           (sample.shape[Backdoor.CHANNEL_FIRST] == 1 or sample.shape[Backdoor.CHANNEL_FIRST] == 3):
            raise ValueError(f"Duplicate channel found, found {sample.shape[Backdoor.CHANNEL_LAST]} at dimension 2 and {sample.shape[Backdoor.CHANNEL_FIRST]} at dimension 0")

        if sample.shape[Backdoor.CHANNEL_LAST] == 1 or sample.shape[Backdoor.CHANNEL_LAST] == 3:
            return Backdoor.CHANNEL_LAST
        elif sample.shape[Backdoor.CHANNEL_FIRST] == 1 or sample.shape[Backdoor.CHANNEL_FIRST] == 3:
            return Backdoor.CHANNEL_FIRST
        else:
            raise ValueError(f"Invalid channel shape, found {sample.shape[Backdoor.CHANNEL_LAST]} at dimension 2 and {sample.shape[Backdoor.CHANNEL_FIRST]} at dimension 0")
        
    def __check_image_size(self, sample: torch.Tensor, channel_loc: int):
        image_size = list(sample.shape)[-3:]
        del image_size[channel_loc]
        return image_size
    
    def get_target(self, type: str, trigger: torch.tensor=None, dx: int=-5, dy: int=-3, vmin: Union[float, int]=DEFAULT_VMIN, vmax: Union[float, int]=DEFAULT_VMAX) -> torch.Tensor:
        channel_loc = self.__check_channel(sample=trigger, channel_first=None)
        channel = trigger.shape[channel_loc]
        image_size = self.__check_image_size(sample=trigger, channel_loc=channel_loc)
        print(f"image size: {image_size}")
        if type == Backdoor.TARGET_TG:
            if trigger == None:
                raise ValueError("trigger shouldn't be none")
            return Backdoor.__bg2grey(trigger.clone().detach(), vmin=vmin, vmax=vmax)
        elif type == Backdoor.TARGET_SHIFT:
            if trigger == None:
                raise ValueError("trigger shouldn't be none")
            return Backdoor.__bg2grey(Backdoor.__roll(trigger.clone().detach(), dx=dx, dy=dy), vmin=vmin, vmax=vmax)
        elif type == Backdoor.TARGET_CORNER:
            b1 = (None, None)
            b2 = (10, 10)
            return Backdoor.__bg2grey(trig=Backdoor.__get_grey_box_trig(b1=b1, b2=b2, channel=channel, image_size=image_size, vmin=vmin, vmax=vmax), vmin=vmin, vmax=vmax)
        elif type == Backdoor.TARGET_SHOE:
            trans = self.__get_transform(channel=channel, image_size=image_size, vmin=vmin, vmax=vmax)
            ds = FashionMNIST(root=self.__root, train=True, download=True, transform=trans)
            return Backdoor.__bg2grey(trig=ds[0][0], vmin=vmin, vmax=vmax)
        # elif type == Backdoor.TARGET_HAT:
        #     return self.__get_img_target(path="static/hat.png", channel=channel, image_size=image_size, vmin=vmin, vmax=vmax)
        elif type == Backdoor.TARGET_HAT:
            return self.__get_img_target(path="static/fedora-hat.png", channel=channel, image_size=image_size, vmin=vmin, vmax=vmax)
        elif type == Backdoor.TARGET_CAT:
            return self.__get_img_target(path=Backdoor.CAT_IMG, channel=channel, image_size=image_size, vmin=vmin, vmax=vmax)
        else:
            raise NotImplementedError(f"Target type {type} isn't found")
        
    def show_image(self, img: torch.Tensor):
        plt.axis('off')        
        plt.tight_layout()
        plt.imshow(img.permute(1, 2, 0).squeeze(), cmap='gray')
        plt.show()
        
class ImagePathDataset(torch.utils.data.Dataset):
    IMAGE_EXTENSIONS = {'bmp', 'jpg', 'jpeg', 'pgm', 'png', 'ppm', 'tif', 'tiff', 'webp'}
    
    def __init__(self, path, transforms=None, njobs: int=-1):
        self.__path = pathlib.Path(path)
        self.__files = sorted([file for ext in ImagePathDataset.IMAGE_EXTENSIONS
                       for file in self.__path.glob('*.{}'.format(ext))])
        self.__transforms = transforms
        self.__njobs = njobs

    def __len__(self):
        return len(self.__files)

    @staticmethod
    # @lru_cache(1000)
    def __read_img(path):
        return transforms.ToTensor()(Image.open(path).copy().convert('RGB'))
    
    def __getitem__(self, i):
        def read_imgs(paths: Union[str, List[str]]):
            trans_ls = [transforms.Lambda(ImagePathDataset.__read_img)]
            if self.__transforms != None:
                trans_ls += self.__transforms
                
            if isinstance(paths, list):
                if self.__njobs == None:
                    imgs = [Compose(trans_ls)(path) for path in paths]
                else:
                    imgs = list(Parallel(n_jobs=self.__njobs)(delayed(Compose(trans_ls))(path) for path in paths))
                return torch.stack(imgs)
            return transforms.ToTensor()(Image.open(paths).convert('RGB'))

        img = Compose([transforms.Lambda(read_imgs)])(self.__files[i])
        return img
            
# %%
if __name__ == "__main__":
    # You can use the following code to visualize the triggers and the targets
    ds_root = os.path.join('datasets')
    dsl = DatasetLoader(root=ds_root, name=DatasetLoader.CIFAR10, batch_size=128).set_poison(trigger_type=Backdoor.TRIGGER_GLASSES, target_type=Backdoor.TARGET_CAT, clean_rate=0.2, poison_rate=0.4).prepare_dataset(mode=DatasetLoader.MODE_FIXED)
    print(f"Full Dataset Len: {len(dsl)}")

    train_ds = dsl.get_dataset()
    sample = train_ds[0]
    print(f"{sample.keys()}")
    print(f"Full Dataset Len: {len(train_ds)} | Sample Len: {len(sample)}")
    print(f"Clean Target: {sample['target'].shape} | Label: {sample['label']}  | pixel_values: {sample['pixel_values'].shape}")
    print(f"Clean PIXEL_VALUES Shape: {sample['pixel_values'].shape} | vmin: {torch.min(sample['pixel_values'])} | vmax: {torch.max(sample['pixel_values'])} | CLEAN: {sample['is_clean']}")
    dsl.show_sample(sample['pixel_values'])
    print(f"Clean TARGET Shape: {sample['target'].shape} | vmin: {torch.min(sample['target'])} | vmax: {torch.max(sample['target'])} | CLEAN: {sample['is_clean']}")
    dsl.show_sample(sample['target'])
    print(f"Clean IMAGE Shape: {sample['image'].shape} | vmin: {torch.min(sample['image'])} | vmax: {torch.max(sample['image'])} | CLEAN: {sample['is_clean']}")
    dsl.show_sample(sample['image'])
    
    # Count clean samples and poison samples
    # is_cleans = torch.tensor(train_ds[:]['is_clean'])
    # print(f"clean_n: {torch.count_nonzero(torch.where(is_cleans, 1, 0))}, poison_n: {torch.count_nonzero(torch.where(is_cleans, 0, 1))}")
    
    # CIFAR10
    # sample = train_ds[36000]
    # sample = train_ds[5000] # for label = 1
    # MNIST
    # sample = train_ds[60000]
    # sample = train_ds[6742]
    # sample = train_ds[3371]
    # sample = train_ds[14000]
    # sample = train_ds[35000] # For FIXED_MODE
    # CELEBA
    # sample = train_ds[101300]
    # CELEBA-HQ
    sample = train_ds[18000] # For FIXED_MODE
    
    noise = torch.randn_like(sample['target'], dtype=torch.float)
    
    print(f"Full Dataset Len: {len(train_ds)} | Sample Len: {len(sample)}")
    print(f"Backdoor Target: {sample['target'].shape} | Label: {sample['label']}  | pixel_values: {sample['pixel_values'].shape}")
    print(f"Backdoor PIXEL_VALUES Shape: {sample['pixel_values'].shape} | vmin: {torch.min(sample['pixel_values'])} | vmax: {torch.max(sample['pixel_values'])} | CLEAN: {sample['is_clean']}")
    dsl.show_sample(sample['pixel_values'])
    print(f"Backdoor TARGET Shape: {sample['target'].shape} | vmin: {torch.min(sample['target'])} | vmax: {torch.max(sample['target'])} | CLEAN: {sample['is_clean']}")
    dsl.show_sample(sample['target'])
    print(f"Backdoor Noisy PIXEL_VALUES Shape: {sample['pixel_values'].shape} | vmin: {torch.min(sample['pixel_values'])} | vmax: {torch.max(sample['pixel_values'])} | CLEAN: {sample['is_clean']}")
    dsl.show_sample(sample['pixel_values'] + noise)
    print(f"Backdoor IMAGE Shape: {sample['image'].shape} | vmin: {torch.min(sample['image'])} | vmax: {torch.max(sample['image'])} | CLEAN: {sample['is_clean']}")
    dsl.show_sample(sample['image'])

    # create dataloader
    train_dl = dsl.get_dataloader()

    batch = next(iter(train_dl))
    
    # Backdoor
    channel = 3
    image_size = dsl.image_size
    grid_size = 5
    vmin = float(0.0)
    vmax = float(1.0)
    run = os.path.dirname(os.path.abspath(__file__))
    root_p = os.path.join(run, 'datasets')
    backdoor = Backdoor(root=root_p)
    
    # BOX_14 Trigger
    tr = backdoor.get_trigger(type=Backdoor.TRIGGER_STOP_SIGN_14, channel=channel, image_size=image_size, vmin=vmin, vmax=vmax)
    backdoor.show_image(img=tr)
    # SM_BOX Trigger
    tr = backdoor.get_trigger(type=Backdoor.TRIGGER_BOX_14, channel=channel, image_size=image_size, vmin=vmin, vmax=vmax)
    backdoor.show_image(img=tr)
    # XSM_BOX Trigger
    tr = backdoor.get_trigger(type=Backdoor.TRIGGER_BOX_11, channel=channel, image_size=image_size, vmin=vmin, vmax=vmax)
    backdoor.show_image(img=tr)
    # XXSM_BOX Trigger
    tr = backdoor.get_trigger(type=Backdoor.TRIGGER_BOX_8, channel=channel, image_size=image_size, vmin=vmin, vmax=vmax)
    backdoor.show_image(img=tr)
    # XXXSM_BOX Trigger
    tr = backdoor.get_trigger(type=Backdoor.TRIGGER_BOX_4, channel=channel, image_size=image_size, vmin=vmin, vmax=vmax)
    backdoor.show_image(img=tr)
    # GLASSES Trigger
    tr = backdoor.get_trigger(type=Backdoor.TRIGGER_GLASSES, channel=3, image_size=image_size, vmin=vmin, vmax=1)
    backdoor.show_image(img=tr)
    # Cat Target
    tg = backdoor.get_target(type=Backdoor.TARGET_CAT, trigger=tr, vmin=vmin, vmax=1)
    backdoor.show_image(img=tg)
    # Hat Target
    tg = backdoor.get_target(type=Backdoor.TARGET_HAT, trigger=tr, vmin=vmin, vmax=1)
    backdoor.show_image(img=tg)
    
# %%
