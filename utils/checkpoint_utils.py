import torch
from torch.distributed._shard.checkpoint import FileSystemReader
import torch.distributed._shard.checkpoint as dist_cp
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    StateDictType,
    FullStateDictConfig,  # general model non-sharded, non-flattened params
)
from torch.distributed.checkpoint.default_planner import (
    DefaultSavePlanner,
)
import time
from datetime import datetime
from pathlib import Path

def get_date_of_run():
    """create date and time for file save uniqueness
    example: 2022-05-07-08:31:12_PM'
    """
    date_of_run = datetime.now().strftime("%Y-%m-%d-%I:%M:%S_%p")
    print(f"--> current date and time of run = {date_of_run}")
    return date_of_run


# create singleton saving policies to avoid making over and over
fullstate_save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)


def save_model_checkpoint(
    model,
    optimizer,
    rank,
    cfg,
    epoch=1,
):
    """saving model via rank0 cpu streaming and full_state_dict"""

    with FSDP.state_dict_type(
        model, StateDictType.FULL_STATE_DICT, fullstate_save_policy
    ):
        cpu_state = model.state_dict()

        print(f"saving process: rank {rank}  done w model state_dict\n")
   

    if rank == 0:
        print(f"--> saving model ...")
        # create save path
        folder_name = (
        cfg.dist_checkpoint_root_folder
        + "/"
        + cfg.dist_checkpoint_folder
        + "-"
        + cfg.model_name
        )
        save_dir = Path.cwd() / folder_name
        save_dir.mkdir(parents=True, exist_ok=True)
        save_name = cfg.model_name + "-" + str(epoch) + ".pt"
        save_full_path = str(save_dir) + "/" + save_name

        # save model
        torch.save(cpu_state, save_full_path)

        
        print(f"model checkpoint saved for epoch {epoch} at {save_full_path}\n")

def save_model_and_optimizer_sharded(model, rank, cfg,optim=None):
    """save model and optimizer via sharded_state_dict to save_dir"""
    
    folder_name = (
        cfg.dist_checkpoint_root_folder
        + "/"
        + cfg.dist_checkpoint_folder
        + "-"
        + cfg.model_name
    )

    save_dir = Path.cwd() / folder_name
    if rank == 0:
        print(f"Saving model to {save_dir}")

    distributed_writer = dist_cp.FileSystemWriter(
        save_dir,
    )
    t0 = time.perf_counter()

    with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT):
        
        state_dict = {"model": model.state_dict()}
        if optim is not None:
            state_dict["optim"] = FSDP.optim_state_dict(model, optim)

        dist_cp.save_state_dict(
            state_dict=state_dict,
            storage_writer=distributed_writer,
            planner=DefaultSavePlanner(),
            
        )
    dist.barrier()
    t1 = time.perf_counter()
    if rank == 0:
        print(f"Sharded state checkpoint saved to {save_dir}")
        print(
            f"Checkpoint Time = {t1-t0:.4f}\n"
        )

def save_optimizer_checkpoint(model, optimizer, rank, cfg, epoch=1):
    """save optimizer state via full state dict"""

   
    print(f"--> optim state call on rank {rank}\n")

    # pull all sharded optimizer states to rank0 cpu...

    optim_state = FSDP.full_optim_state_dict(model, optimizer)

    
    print(f"optim state dict ready on {rank} and len of {len(optim_state)}\n")

    if rank == 0:
        folder_name = (
        cfg.dist_checkpoint_root_folder
        + "/"
        + cfg.dist_checkpoint_folder
        + "-"
        + cfg.model_name
        )
        save_dir = Path.cwd() / folder_name
        save_dir.mkdir(parents=True, exist_ok=True)

        opt_save_name = (
            "optimizer" + "-" + cfg.model_name + "-" + str(epoch) + ".pt"
        )
        opt_save_full_path = save_dir / opt_save_name

        print(f"--> saving optimizer state...")

        torch.save(optim_state, opt_save_full_path)

        print(f"--> saved {opt_save_full_path} to disk")

def load_sharded_split_single_gpu(model,ctx_model, question_model, model_path):    
    state_dict = {
        "model": model.state_dict()
    }
    ctx_state_dict = {"model": {}}    
    question_state_dict = {"model": {}}    
    dist_cp.load_state_dict(
                state_dict=state_dict,
                storage_reader= FileSystemReader(model_path),
                no_dist=True,
            )

    for k, v in state_dict["model"].items():
        if k.startswith("ctx_encoder."):
            # print("ctx key: ", k)
            ctx_state_dict["model"][k.replace("ctx_encoder.", "")] = v
        else:
            # print("question key: ", k)
            question_state_dict["model"][k] = v
    ctx_model.load_state_dict(ctx_state_dict["model"])
    question_model.load_state_dict(question_state_dict["model"])
    print(f"Sharded ctx state checkpoint loaded from {model_path}")
    return ctx_model, question_model
