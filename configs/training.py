# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.
from dataclasses import dataclass
from typing import ClassVar


@dataclass
class training_config:
    model_name: str
    token_name: str
    enable_fsdp: bool
    low_cpu_fsdp: bool
    run_validation: bool
    batch_size_training: int
    num_epochs: int
    num_workers_dataloader: int
    weight_decay: float
    gamma: float
    seed: int
    use_fp16: bool
    mixed_precision: bool
    val_batch_size: int
    micro_batch_size: int
    ctx_use_peft: bool
    model_use_peft: bool
    output_dir: str
    freeze_layers: bool
    num_freeze_layers: int
    quantization: bool
    one_gpu: bool
    save_model: bool
    dist_checkpoint_root_folder: str# will be used if using FSDP
    dist_checkpoint_folder: str# will be used if using FSDP
    ret_checkpoint_folder: str# will be used if using FSDP
    retriever: str
    save_optimizer: bool # will be used if using FSDP
    use_fast_kernels: bool# Enable using SDPA from PyTroch Accelerated Transformers, make use Flash Attention and Xformer memory-efficient kernels
    peft_method: str# None , llama_adapter, prefix    
    lr: float
    memory_bank_length: int
    dataset: str# dpr_nq_dataset, dpr_nq_hardneg_dataset
    single: bool
    all_gather: bool
    add_vocab: bool
    freeze_question_encoder: bool = False
    freeze_ctx_encoder: bool = False
    natural_form: bool = True
    cpu_np_head: bool = True
    load_np_head: bool = False
    train: bool = True
    compare: bool = False
    ctx_proj_layer: bool = False
    question_proj_layer: bool = False
    ret_first: bool = False
    add_ctxemb: bool = False
    np_weight : float = 100.0
    clipping_norm: float = -1.0
    resume_epoch: int = 0