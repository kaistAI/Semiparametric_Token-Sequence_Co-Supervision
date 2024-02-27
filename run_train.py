import fire
import os, torch
import torch.distributed as dist
import torch.optim as optim
import json
from pkg_resources import packaging

from configs import fsdp_config, training_config
from peft import get_peft_model, prepare_model_for_int8_training
from utils.fsdp_utils import fsdp_auto_wrap_policy
from utils.convert_utils import save_to_hf
from utils.config_utils import (
    update_config,
    generate_dataset_config,
)
from utils.setup_utils import (
    setup,
    setup_environ_flags,
    clear_gpu_cache,
    get_policies,
)
from transformers import LlamaConfig, LlamaTokenizer, LlamaModel
from transformers.models.llama.modeling_llama import LlamaDecoderLayer

from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
)
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DistributedSampler
from policies.activation_checkpointing_functions import apply_fsdp_checkpointing
from policies.anyprecision_optimizer import AnyPrecisionAdamW

from model.joint_modeling_llama import JointModel
from dataset.generate_dataset import train_data_module
from utils.train_utils import train

def main(**kwargs):
    with open(kwargs["training_argument"], "r") as f:
        json_obj = json.load(f)
    train_config = training_config(**json_obj)
    try:
        import wandb
        wandb_log = wandb.init(project="sentence-decoding")
    except:
        wandb_log = None
    update_config((train_config, fsdp_config), **kwargs)
    if train_config.resume_epoch>0:
        resume_model_ckpt_path = os.path.join(
            train_config.dist_checkpoint_root_folder,
            f"{train_config.dist_checkpoint_folder}_epoch{train_config.resume_epoch-1}" + "-" + train_config.model_name,
        )
        assert os.path.isdir(resume_model_ckpt_path)
    torch.cuda.manual_seed(train_config.seed)
    torch.manual_seed(train_config.seed)
    if train_config.enable_fsdp:
        setup()
        # torchrun specific
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = int(os.environ["RANK"])

    if torch.distributed.is_initialized():
        torch.cuda.set_device(local_rank)
        clear_gpu_cache(local_rank)
        setup_environ_flags(rank)
    dataset_config = generate_dataset_config(train_config, kwargs)
    gradient_accumulation_steps = (
        train_config.batch_size_training // train_config.micro_batch_size
    )
    if train_config.resume_epoch>0:
        llama_config = LlamaConfig.from_pretrained(os.path.join(resume_model_ckpt_path, "ctx"))
    else:
        llama_config = LlamaConfig.from_pretrained(train_config.model_name)
    train_tokenizer = LlamaTokenizer.from_pretrained(
        train_config.token_name, model_max_length=1024
    )
    train_tokenizer.add_special_tokens(dict(pad_token="<PAD>"))
    if train_config.natural_form:
        train_tokenizer.add_special_tokens(
            {"additional_special_tokens": ["[Cs]", "[Ce]"]}
        )
        train_tokenizer.add_special_tokens({"additional_special_tokens": ["[Ret]"]})
    if train_config.resume_epoch>0:
        save_to_hf(train_config, resume_model_ckpt_path, kwargs, rank)
    if train_config.enable_fsdp and train_config.low_cpu_fsdp:
        """
        for FSDP, we can save cpu memory by loading pretrained model on rank0 only.
        this avoids cpu oom when loading large models like llama 70B, in which case
        model alone would consume 2+TB cpu mem (70 * 4 * 8). This will add some comms
        overhead and currently requires latest nightly.
        """
        v = packaging.version.parse(torch.__version__)
        verify_latest_nightly = v.is_devrelease and v.dev >= 20230701
        if not verify_latest_nightly:
            raise Exception(
                "latest pytorch nightly build is required to run with low_cpu_fsdp config, "
                "please install latest nightly."
            )

        if rank == 0:
            model = JointModel.from_pretrained(
                os.path.join(resume_model_ckpt_path, "question") if train_config.resume_epoch>0 else train_config.model_name,
                train_config,
                load_in_8bit=True if train_config.quantization else None,
                device_map="auto" if train_config.quantization else None,
            )
            if train_config.resume_epoch>0:
                model.ctx_encoder.resize_token_embeddings(len(train_tokenizer))
            model.ctx_encoder = LlamaModel.from_pretrained(
                os.path.join(resume_model_ckpt_path, "ctx") if train_config.resume_epoch>0 else train_config.model_name,
                config=llama_config,
                load_in_8bit=True if train_config.quantization else None,
                device_map="auto" if train_config.quantization else None,
            )
        else:
            with torch.device("meta"):
                model = JointModel(
                    llama_config,
                    train_config=train_config,
                    # vanilla_vocab_size=vanilla_vocab_size,
                    #    tokenizer= train_tokenizer
                )
    else:
        model = JointModel.from_pretrained(
            os.path.join(resume_model_ckpt_path, "question") if train_config.resume_epoch>0 else train_config.model_name,
            train_config,
            # vanilla_vocab_size=vanilla_vocab_size,
            # tokenizer= train_tokenizer,
            load_in_8bit=True if train_config.quantization else None,
            device_map="auto" if train_config.quantization else None,
        )
        if train_config.resume_epoch>0:
            model.ctx_encoder.resize_token_embeddings(len(train_tokenizer))

        model.ctx_encoder = LlamaModel.from_pretrained(
            os.path.join(resume_model_ckpt_path, "ctx") if train_config.resume_epoch>0 else train_config.model_name,
            config=llama_config,
            load_in_8bit=True if train_config.quantization else None,
            device_map="auto" if train_config.quantization else None,
        )
    model.set_vanilla_vocab_size(len(train_tokenizer))
    if train_config.enable_fsdp and train_config.use_fast_kernels:
        """
        For FSDP and FSDP+PEFT, setting 'use_fast_kernels' will enable
        using of Flash Attention or Xformer memory-efficient kernels
        based on the hardware being used. This would speed up fine-tuning.
        """
        try:
            from optimum.bettertransformer import BetterTransformer

            model = BetterTransformer.transform(model)
        except ImportError:
            print(
                "Module 'optimum' not found. Please install 'optimum' it before proceeding."
            )
    # Prepare the model for int8 training if quantization is enabled
    if train_config.quantization:
        model = prepare_model_for_int8_training(model)

    # Convert the model to bfloat16 if fsdp and pure_bf16 is enabled
    if train_config.enable_fsdp and fsdp_config.pure_bf16:
        model.to(torch.bfloat16)
    # setting up FSDP if enable_fsdp is enabled
    if train_config.enable_fsdp:
        mixed_precision_policy, wrapping_policy = get_policies(fsdp_config, rank)
        my_auto_wrapping_policy = fsdp_auto_wrap_policy(model, LlamaDecoderLayer)
        model = FSDP(
            model,
            auto_wrap_policy=my_auto_wrapping_policy
            if (train_config.ctx_use_peft or train_config.model_use_peft )
            else wrapping_policy,
            mixed_precision=mixed_precision_policy
            if not fsdp_config.pure_bf16
            else None,
            sharding_strategy=fsdp_config.sharding_strategy,
            device_id=torch.cuda.current_device(),
            limit_all_gathers=True,
            sync_module_states=train_config.low_cpu_fsdp,
            param_init_fn=lambda module: module.to_empty(
                device=torch.device("cuda"), recurse=False
            )
            if train_config.low_cpu_fsdp and rank != 0
            else None,
        )
        if fsdp_config.fsdp_activation_checkpointing:
            apply_fsdp_checkpointing(model)
    elif not train_config.quantization and not train_config.enable_fsdp:
        model.to("cuda")
    # Load and preprocess the dataset for training and validation
    data_module_train = train_data_module(
        ctx_tokenizer=train_tokenizer,
        question_tokenizer=train_tokenizer,
        dataset_config=dataset_config,
    )
    dataset_train = data_module_train["train_dataset"]
    data_collator_train = data_module_train["data_collator"]
    if not train_config.enable_fsdp or rank == 0:
        print(f"--> Training Set Length = {len(dataset_train)}")
    train_sampler = None
    if train_config.enable_fsdp:
        train_sampler = DistributedSampler(
            dataset_train,
            rank=dist.get_rank(),
            num_replicas=dist.get_world_size(),
            shuffle=True,
        )
    train_dataloader = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=train_config.batch_size_training,
        num_workers=train_config.num_workers_dataloader,
        pin_memory=True,
        sampler=train_sampler if train_sampler else None,
        drop_last=True,
        collate_fn=data_collator_train,
    )
    if fsdp_config.pure_bf16 and fsdp_config.optimizer == "anyprecision":
        optimizer = AnyPrecisionAdamW(
            model.parameters(),
            lr=train_config.lr * (train_config.gamma ** train_config.resume_epoch),
            momentum_dtype=torch.bfloat16,
            variance_dtype=torch.bfloat16,
            use_kahan_summation=False,
        )
    else:
        optimizer = optim.AdamW(
            model.parameters(),
            lr=train_config.lr * (train_config.gamma ** train_config.resume_epoch),
            weight_decay=0.0,
        )
    scheduler = StepLR(optimizer, step_size=1, gamma=train_config.gamma)
    wandb.watch(model)

    results = train(
        model,
        train_dataloader,
        optimizer,
        scheduler,
        gradient_accumulation_steps,
        train_config,
        fsdp_config if train_config.enable_fsdp else None,
        local_rank if train_config.enable_fsdp else None,
        rank if train_config.enable_fsdp else None,
        clipping_norm=train_config.clipping_norm,
        wandb_log=wandb_log
    )
    model_ckpt_path = os.path.join(
        train_config.dist_checkpoint_root_folder,
        train_config.dist_checkpoint_folder + "-" + train_config.model_name,
    )
    save_to_hf(train_config, model_ckpt_path, kwargs, rank)

if __name__ == "__main__":
    fire.Fire(main)
