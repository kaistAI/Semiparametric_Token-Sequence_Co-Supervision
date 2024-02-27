import os
import sys
import yaml
import utils.checkpoint_utils as model_checkpointing

from torch.nn import functional as F
from torch.distributed.fsdp import StateDictType
import torch.distributed as dist
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))



def save_sole(model, train_config, fsdp_config, optimizer, epoch, rank):
    folder_name = (
        train_config.dist_checkpoint_root_folder
        + "/"
        + train_config.dist_checkpoint_folder
        + "-"
        + train_config.model_name
    )
    if train_config.enable_fsdp:
        dist.barrier()
    if train_config.model_use_peft:
        if train_config.enable_fsdp:
            if rank == 0:
                print(f"we are about to save the PEFT modules")
        else:
            print(f"we are about to save the PEFT modules")
        model.save_pretrained(folder_name)
        if train_config.enable_fsdp:
            if rank == 0:
                print(
                    f"PEFT modules are saved in {folder_name} directory"
                )
        else:
            print(
                f"PEFT modules are saved in {folder_name} directory"
            )

    else:
        if (
            not train_config.model_use_peft
            and fsdp_config.checkpoint_type == StateDictType.FULL_STATE_DICT
        ):
            model_checkpointing.save_model_checkpoint(
                model, optimizer, rank, train_config, epoch=epoch
            )
        elif (
            not train_config.model_use_peft
            and fsdp_config.checkpoint_type
            == StateDictType.SHARDED_STATE_DICT
        ):
            print(
                " Saving the FSDP model checkpoints using SHARDED_STATE_DICT"
            )
            print("=====================================================")

            model_checkpointing.save_model_and_optimizer_sharded(
                model, rank, train_config
            )
            if train_config.save_optimizer:
                model_checkpointing.save_model_and_optimizer_sharded(
                    model, rank, train_config, optim=optimizer
                )
                print(
                    " Saving the FSDP model checkpoints and optimizer using SHARDED_STATE_DICT"
                )
                print(
                    "====================================================="
                )

        if not train_config.model_use_peft and train_config.save_optimizer:
            model_checkpointing.save_optimizer_checkpoint(
                model, optimizer, rank, train_config, epoch=epoch
            )
            print(
                " Saving the FSDP model checkpoints and optimizer using FULL_STATE_DICT"
            )
            print("=====================================================")
    if train_config.enable_fsdp:
        dist.barrier()

def save_joint(model, train_config, fsdp_config, optimizer, epoch, rank):
    folder_name = (
        train_config.dist_checkpoint_root_folder
        + "/"
        + train_config.dist_checkpoint_folder
        + "-"
        + train_config.model_name
    )
    if train_config.enable_fsdp:
        dist.barrier()
    if train_config.ctx_use_peft and train_config.model_use_peft:
        if train_config.enable_fsdp:
            if rank == 0:
                print(f"we are about to save the PEFT modules")
        else:
            print(f"we are about to save the PEFT modules")
        model.ctx_encoder.save_pretrained(folder_name + "/" + "ctx")
        model.model.save_pretrained(folder_name + "/" + "question")
        if train_config.enable_fsdp:
            if rank == 0:
                print(
                    f"PEFT modules are saved in {folder_name} directory"
                )
        else:
            print(
                f"PEFT modules are saved in {folder_name} directory"
            )

    else:
        if train_config.ctx_use_peft:
            model.ctx_encoder.save_pretrained(folder_name + "/" + "ctx")
        if train_config.model_use_peft:
            model.model.save_pretrained(folder_name + "/" + "question")
        if not (train_config.ctx_use_peft or train_config.model_use_peft):
            if train_config.ctx_use_peft:
                save_model = model.model
            elif train_config.model_use_peft:
                save_model = model.ctx_encoder
            else:
                save_model = model
            if fsdp_config.checkpoint_type == StateDictType.FULL_STATE_DICT:
                model_checkpointing.save_model_checkpoint(
                    save_model, optimizer, rank, train_config, epoch=epoch
                )
            elif fsdp_config.checkpoint_type == StateDictType.SHARDED_STATE_DICT:
                print(
                    " Saving the FSDP model checkpoints using SHARDED_STATE_DICT"
                )
                print("=====================================================")

                model_checkpointing.save_model_and_optimizer_sharded(
                    save_model, rank, train_config
                )
                if train_config.save_optimizer:
                    model_checkpointing.save_model_and_optimizer_sharded(
                        save_model, rank, train_config, optim=optimizer
                    )
                    print(
                        " Saving the FSDP model checkpoints and optimizer using SHARDED_STATE_DICT"
                    )
                    print(
                        "====================================================="
                    )                        
            else:
                
                if train_config.save_optimizer:
                    model_checkpointing.save_optimizer_checkpoint(
                        save_model, optimizer, rank, train_config, epoch=epoch
                    )
                    print(
                        " Saving the FSDP model checkpoints and optimizer using FULL_STATE_DICT"
                    )
                    print("=====================================================")

    if train_config.enable_fsdp:
        dist.barrier()
        
def save_train_params(train_config, fsdp_config, rank):
    """
    This function saves the train_config and FSDP config into a train_params.yaml.
    This will be used by converter script in the inference folder to fetch the HF model name or path.
    It also would be hepful as a log for future references.
    """
    # Convert the train_config and fsdp_config objects to dictionaries, 
    # converting all values to strings to ensure they can be serialized into a YAML file
    train_config_dict = {k: str(v) for k, v in vars(train_config).items() if not k.startswith('__')}
    fsdp_config_dict = {k: str(v) for k, v in vars(fsdp_config).items() if not k.startswith('__')}
    # Merge the two dictionaries into one
    train_params_dict = {**train_config_dict, **fsdp_config_dict}
    # Construct the folder name (follwoing FSDP checkpointing style) using properties of the train_config object
    folder_name = (
    train_config.dist_checkpoint_root_folder
    + "/"
    + train_config.dist_checkpoint_folder
    + "-"
    + train_config.model_name
    )

    save_dir = Path.cwd() / folder_name
    # If the directory does not exist, create it
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # Convert the dictionary to a YAML string
    config_yaml = yaml.dump(train_params_dict, indent=4)
    file_name = os.path.join(save_dir,'train_params.yaml')

    # Check if there's a directory with the same name as the file
    if os.path.isdir(file_name):
        print(f"Error: {file_name} is a directory, not a file.")
    else:
        # Write the YAML string to the file
        with open(file_name, 'w') as f:
            f.write(config_yaml)
        if rank==0:
            print(f"training params are saved in {file_name}")