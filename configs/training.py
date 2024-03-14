# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.
from dataclasses import dataclass
from typing import ClassVar

@dataclass
class training_config:
    seed: int = 2
    model_name: str = "meta-llama/Llama-2-7b-hf"
    token_name: str = "meta-llama/Llama-2-7b-hf"
    num_epochs: int = 3
    lr: float = 2e-5
    gamma: float = 0.85
    dist_checkpoint_root_folder: str = "model_checkpoints"
    dist_checkpoint_folder: str = "ntp_nsp_cosupervision"
    enable_fsdp: bool = True
    batch_size_training: int = 8
    micro_batch_size: int = 8
    low_cpu_fsdp: bool = True
    quantization: bool = False
    use_fast_kernels: bool = False
    num_workers_dataloader: int = 1
    use_fp16: bool = False
    save_model: bool = True
    save_optimizer: bool = False
    dataset: str = "selfrag_multi_dataset"
    single: bool = False
    all_gather: bool = True
    np_weight: float = 100.0
    clipping_norm: float = 1.0
    resume_epoch: int = 0