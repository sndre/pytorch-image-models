# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import datetime
import gc
import inspect
import logging
import pandas as pd
import pathlib
from typing import TypeVar

import torch
import os

from timm.data import create_loader
from torch.utils.data import WeightedRandomSampler, RandomSampler

def create_file_handler(log_dir: str) -> logging.FileHandler:
    path = pathlib.Path.cwd() / log_dir / f'{datetime.datetime.now():log_%Y-%m-%d-%H-%M-%S}.log'
    path.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(path, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        '%(asctime)s.%(msecs)04d\t%(levelname)s\t%(name)s\t%(message)s', datefmt='%Y-%m-%dT%H:%M:%S'
    )
    file_handler.setFormatter(formatter)
    return file_handler


def configure_logging(
    log_to_console: bool = True,
    log_to_file: bool = True,
    log_dir: str = 'log',
    level: int = logging.INFO,
) -> None:
    handlers: list[logging.Handler] = []

    if log_to_console:
        handler = logging.StreamHandler()
        handler.setLevel(level)
        formatter = logging.Formatter('%(message)s')
        handler.setFormatter(formatter)
        handlers.append(handler)

    if log_to_file:
        handlers.append(create_file_handler(log_dir=log_dir))

    logging.basicConfig(
        handlers=handlers,
        level=logging.NOTSET,
    )


def cleanup_memory() -> None:
    """Run GC and clear GPU memory."""
    caller_name = ''
    try:
        caller_name = f' (from {inspect.stack()[1].function})'
    except (ValueError, KeyError):
        pass

    def total_reserved_mem() -> int:
        return sum(torch.cuda.memory_reserved(device=i) for i in range(torch.cuda.device_count()))

    memory_before = total_reserved_mem()

    # gc.collect and empty cache are necessary to clean up GPU memory if the model was distributed
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        memory_after = total_reserved_mem()
        logging.debug(
            f"GPU memory{caller_name}: {memory_before / (1024 ** 3):.2f} -> {memory_after / (1024 ** 3):.2f} GB"
            f" ({(memory_after - memory_before) / (1024 ** 3):.2f} GB)"
        )


T = TypeVar('T')


def map_tensors(obj: T, device: torch.device | str | None = None, dtype: torch.dtype | None = None) -> T:
    """Recursively map tensors to device and dtype."""
    if isinstance(obj, torch.Tensor):
        if device is not None:
            obj = obj.to(device=device)
        if dtype is not None:
            obj = obj.to(dtype=dtype)
        return obj
    elif isinstance(obj, (list, tuple)):
        return type(obj)(map_tensors(x, device, dtype) for x in obj)
    elif isinstance(obj, dict):
        return {k: map_tensors(v, device, dtype) for k, v in obj.items()}  # type: ignore
    else:
        return obj

class TensorFile:
    def __init__(self, file_template):
        self.file_template = file_template

    def save(self, tensor, tensor_name, idx):
        file_name = self.file_template.format(tensor_name, idx)
        os.makedirs(os.path.dirname(file_name), exist_ok=True)
        torch.save(tensor, file_name)

    def load(self, tensor_name, idx):
        file_name = self.file_template.format(tensor_name, idx)
        if os.path.exists(file_name):
            return torch.load(file_name)
        else:
            return None

def create_stratified_loader(dataset, metadata_path, data_config, args, device=torch.device("cuda")):
    # Count the number of classes and instances per class
    df = pd.read_csv(metadata_path)
    class_count = df['label'].value_counts().to_dict()
    
    # Create weights for each instance
    weights = [1.0 / class_count[label] for _, _, label in df.itertuples()]
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

    # Use timm's create_loader to integrate the sampler
    loader = create_loader(
        dataset,
        input_size=data_config['input_size'],
        batch_size=args.batch_size,
        use_prefetcher=args.prefetcher,
        interpolation=data_config['interpolation'],
        mean=data_config['mean'],
        std=data_config['std'],
        num_workers=args.workers,
        sampler=sampler,  # integrate our custom sampler
        crop_pct=0.9,
        crop_mode=data_config['crop_mode'],
        crop_border_pixels=args.crop_border_pixels,
        pin_memory=args.pin_mem,
        device=device,
        tf_preprocessing=args.tf_preprocessing,
    )
    return loader

def create_random_loader(dataset, data_config, args, device=torch.device("cuda")):
    sampler = RandomSampler(dataset)

    # Use timm's create_loader to integrate the sampler
    loader = create_loader(
        dataset,
        input_size=data_config['input_size'],
        batch_size=args.batch_size,
        use_prefetcher=args.prefetcher,
        interpolation=data_config['interpolation'],
        mean=data_config['mean'],
        std=data_config['std'],
        num_workers=args.workers,
        sampler=sampler,  # integrate our custom sampler
        crop_pct=0.9,
        crop_mode=data_config['crop_mode'],
        crop_border_pixels=args.crop_border_pixels,
        pin_memory=args.pin_mem,
        device=device,
        tf_preprocessing=args.tf_preprocessing,
    )
    return loader

def create_loader_with_sampler(dataset, sampler, data_config, args, device=torch.device("cuda")):
    return create_loader(
        dataset,
        input_size=data_config['input_size'],
        batch_size=args.batch_size,
        use_prefetcher=args.prefetcher,
        interpolation=data_config['interpolation'],
        mean=data_config['mean'],
        std=data_config['std'],
        num_workers=args.workers,
        sampler=sampler,  # integrate our custom sampler
        crop_pct=0.9,
        crop_mode=data_config['crop_mode'],
        crop_border_pixels=args.crop_border_pixels,
        pin_memory=args.pin_mem,
        device=device,
        tf_preprocessing=args.tf_preprocessing,
    )
    return loader