from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
import os, torch
from tqdm import tqdm
import time
import torch.distributed as dist
from utils.memory_utils import MemoryTrace
from utils.save_utils import save_train_params, save_joint, save_sole
def train(
    model,
    train_dataloader,
    optimizer,
    lr_scheduler,
    gradient_accumulation_steps,
    train_config,
    fsdp_config=None,
    local_rank=None,
    rank=None,
    clipping_norm=-1.0,
    wandb_log=None
):
    """
    Trains the model on the given dataloader

    Args:
        model: The model to be trained
        train_dataloader: The dataloader containing the training data
        optimizer: The optimizer used for training
        lr_scheduler: The learning rate scheduler
        gradient_accumulation_steps: The number of steps to accumulate gradients before performing a backward/update operation
        num_epochs: The number of epochs to train for
        local_rank: The rank of the current node in a distributed setting
        train_config: The training configuration
        eval_dataloader: The dataloader containing the eval data
        tokenizer: tokenizer used in the eval for decoding the predicitons

    Returns: results dictionary containing average training and validation perplexity and loss
    """
    # Create a gradient scaler for fp16
    if train_config.use_fp16 and train_config.enable_fsdp:
        scaler = ShardedGradScaler()
    elif train_config.use_fp16 and not train_config.enable_fsdp:
        scaler = torch.cuda.amp.GradScaler()
    if train_config.enable_fsdp:
        world_size = int(os.environ["WORLD_SIZE"])
    train_prep = []
    train_loss = []
    epoch_times = []
    checkpoint_times = []
    results = {}
    for epoch in range(train_config.resume_epoch, train_config.num_epochs):
        epoch_start_time = time.perf_counter()
        with MemoryTrace() as memtrace:  # track the memory usage
            model.train()
            total_loss = 0.0

            for step, batch in enumerate(
                tqdm(train_dataloader, colour="blue", desc=f"Training Epoch{epoch}")
            ):
                for key in batch.keys():
                    if torch.is_tensor(batch[key]):
                        if train_config.enable_fsdp:
                            batch[key] = batch[key].to(local_rank)
                        else:
                            batch[key] = batch[key].to("cuda:0")
                loss = model(**batch).loss
                loss /= gradient_accumulation_steps
                float_loss = loss.detach().float()
                total_loss += float_loss
                
                if wandb_log is not None:
                    wandb_log.log({ "step.total_loss": float_loss})
                print(f"device {torch.cuda.current_device()} | total loss: {float_loss}")

                
                if train_config.use_fp16:
                    # if fp16 is enabled, use gradient scaler to handle gradient update
                    scaler.scale(loss).backward()
                    if (step + 1) % gradient_accumulation_steps == 0 or step == len(
                        train_dataloader
                    ) - 1:
                        if clipping_norm>0:
                            scaler.unscale_(optimizer)
                            torch.nn.utils.clip_grad_norm_(model.model.parameters(), clipping_norm)

                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad()
                else:
                    # regular backpropagation when fp16 is not used
                    loss.backward()
                    if (step + 1) % gradient_accumulation_steps == 0 or step == len(
                        train_dataloader
                    ) - 1:
                        if clipping_norm>0:
                            torch.nn.utils.clip_grad_norm_(model.model.parameters(), clipping_norm)

                        optimizer.step()
                        optimizer.zero_grad()
                if train_config.enable_fsdp:
                    if rank == 0:
                        print(
                            f"\n step {step} is completed and loss is {float_loss}"
                        )
                else:
                    print(
                        f"\n step {step} is completed and loss is {float_loss}"
                    )
        epoch_end_time = time.perf_counter() - epoch_start_time
        epoch_times.append(epoch_end_time)
        # Reducing total_loss across all devices if there's more than one CUDA device
        if torch.cuda.device_count() > 1 and train_config.enable_fsdp:
            dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
        train_epoch_loss = total_loss / len(train_dataloader)
        if train_config.enable_fsdp:
            train_epoch_loss = train_epoch_loss / world_size
        train_perplexity = torch.exp(train_epoch_loss)

        train_prep.append(train_perplexity)
        train_loss.append(train_epoch_loss)
        if wandb_log is not None:
            wandb_log.log({"epoch.train_loss": train_epoch_loss, "epoch.train_perplexity": train_perplexity})
        
        if "_epoch" in train_config.dist_checkpoint_folder:
            train_config.dist_checkpoint_folder = train_config.dist_checkpoint_folder.split("_epoch")[0]
        train_config.dist_checkpoint_folder = f"{train_config.dist_checkpoint_folder}_epoch{epoch}" 
        print(f"Save CKPT.. {train_config.dist_checkpoint_folder}")
        
        if train_config.enable_fsdp:
            if rank == 0:
                print(f"Max CUDA memory allocated was {memtrace.peak} GB")
                print(f"Max CUDA memory reserved was {memtrace.max_reserved} GB")
                print(f"Peak active CUDA memory was {memtrace.peak_active_gb} GB")
                print(f"Cuda Malloc retires : {memtrace.cuda_malloc_retires}")
                print(
                    f"CPU Total Peak Memory consumed during the train (max): {memtrace.cpu_peaked + memtrace.cpu_begin} GB"
                )
        else:
            print(f"Max CUDA memory allocated was {memtrace.peak} GB")
            print(f"Max CUDA memory reserved was {memtrace.max_reserved} GB")
            print(f"Peak active CUDA memory was {memtrace.peak_active_gb} GB")
            print(f"Cuda Malloc retires : {memtrace.cuda_malloc_retires}")
            print(
                f"CPU Total Peak Memory consumed during the train (max): {memtrace.cpu_peaked + memtrace.cpu_begin} GB"
            )

        # Update the learning rate as needed
        lr_scheduler.step()

        checkpoint_start_time = time.perf_counter()
        if train_config.save_model:
            save_joint(model, train_config, fsdp_config, optimizer, epoch, rank)
        checkpoint_end_time = time.perf_counter() - checkpoint_start_time
        checkpoint_times.append(checkpoint_end_time)
        # saving the training params including fsdp setting for reference.
        if train_config.enable_fsdp:
            save_train_params(train_config, fsdp_config, rank)

        if train_config.enable_fsdp:
            if rank == 0:
                print(
                    f"Epoch {epoch+1}: train_perplexity={train_perplexity:.4f}, train_epoch_loss={train_epoch_loss:.4f}, epcoh time {epoch_end_time}s"
                )
        else:
            print(
                f"Epoch {epoch+1}: train_perplexity={train_perplexity:.4f}, train_epoch_loss={train_epoch_loss:.4f}, epcoh time {epoch_end_time}s"
            )
    avg_epoch_time = sum(epoch_times) / len(epoch_times)
    avg_checkpoint_time = sum(checkpoint_times) / len(checkpoint_times)
    avg_train_prep = sum(train_prep) / len(train_prep)
    avg_train_loss = sum(train_loss) / len(train_loss)

    results["avg_train_prep"] = avg_train_prep
    results["avg_train_loss"] = avg_train_loss
    results["avg_epoch_time"] = avg_epoch_time
    results["avg_checkpoint_time"] = avg_checkpoint_time

    # saving the training params including fsdp setting for reference.
    if train_config.enable_fsdp:
        save_train_params(train_config, fsdp_config, rank)

    return results


def train_genonly(
    model,
    train_dataloader,
    optimizer,
    lr_scheduler,
    gradient_accumulation_steps,
    train_config,
    fsdp_config=None,
    local_rank=None,
    rank=None,
    clipping_norm=-1.0,
    wandb_log=None
):
    """
    Trains the model on the given dataloader

    Args:
        model: The model to be trained
        train_dataloader: The dataloader containing the training data
        optimizer: The optimizer used for training
        lr_scheduler: The learning rate scheduler
        gradient_accumulation_steps: The number of steps to accumulate gradients before performing a backward/update operation
        num_epochs: The number of epochs to train for
        local_rank: The rank of the current node in a distributed setting
        train_config: The training configuration
        eval_dataloader: The dataloader containing the eval data
        tokenizer: tokenizer used in the eval for decoding the predicitons

    Returns: results dictionary containing average training and validation perplexity and loss
    """
    # Create a gradient scaler for fp16
    if train_config.use_fp16 and train_config.enable_fsdp:
        scaler = ShardedGradScaler()
    elif train_config.use_fp16 and not train_config.enable_fsdp:
        scaler = torch.cuda.amp.GradScaler()
    if train_config.enable_fsdp:
        world_size = int(os.environ["WORLD_SIZE"])
    train_prep = []
    train_loss = []
    epoch_times = []
    checkpoint_times = []
    results = {}
    for epoch in range(train_config.resume_epoch, train_config.num_epochs):
        epoch_start_time = time.perf_counter()
        with MemoryTrace() as memtrace:  # track the memory usage
            model.train()
            total_loss = 0.0

            for step, batch in enumerate(
                tqdm(train_dataloader, colour="blue", desc=f"Training Epoch{epoch}")
            ):
                for key in batch.keys():
                    if torch.is_tensor(batch[key]):
                        # print(key, batch[key])
                        if train_config.enable_fsdp:
                            batch[key] = batch[key].to(local_rank)
                        else:
                            batch[key] = batch[key].to("cuda:0")
                loss = model(
                    input_ids=batch["input_ids"],
                    labels=batch["labels"],
                    attention_mask=batch["attention_mask"],
                ).loss
                loss = loss / gradient_accumulation_steps
                total_loss += loss.detach().float()
                if train_config.use_fp16:
                    # if fp16 is enabled, use gradient scaler to handle gradient update
                    scaler.scale(loss).backward()
                    if (step + 1) % gradient_accumulation_steps == 0 or step == len(
                        train_dataloader
                    ) - 1:
                        if clipping_norm>0:
                            scaler.unscale_(optimizer)
                            torch.nn.utils.clip_grad_norm_(model.model.parameters(), clipping_norm)

                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad()
                else:
                    # regular backpropagation when fp16 is not used
                    loss.backward()
                    if (step + 1) % gradient_accumulation_steps == 0 or step == len(
                        train_dataloader
                    ) - 1:
                        if clipping_norm>0:
                            torch.nn.utils.clip_grad_norm_(model.model.parameters(), clipping_norm)

                        optimizer.step()
                        optimizer.zero_grad()
                if train_config.enable_fsdp:
                    if rank == 0:
                        print(
                            f"\n step {step} is completed and loss is {loss.detach().float()}"
                        )
                else:
                    print(
                        f"\n step {step} is completed and loss is {loss.detach().float()}"
                    )
        epoch_end_time = time.perf_counter() - epoch_start_time
        epoch_times.append(epoch_end_time)
        # Reducing total_loss across all devices if there's more than one CUDA device
        if torch.cuda.device_count() > 1 and train_config.enable_fsdp:
            dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
        train_epoch_loss = total_loss / len(train_dataloader)
        if train_config.enable_fsdp:
            train_epoch_loss = train_epoch_loss / world_size
        train_perplexity = torch.exp(train_epoch_loss)

        train_prep.append(train_perplexity)
        train_loss.append(train_epoch_loss)

        if train_config.enable_fsdp:
            if rank == 0:
                print(f"Max CUDA memory allocated was {memtrace.peak} GB")
                print(f"Max CUDA memory reserved was {memtrace.max_reserved} GB")
                print(f"Peak active CUDA memory was {memtrace.peak_active_gb} GB")
                print(f"Cuda Malloc retires : {memtrace.cuda_malloc_retires}")
                print(
                    f"CPU Total Peak Memory consumed during the train (max): {memtrace.cpu_peaked + memtrace.cpu_begin} GB"
                )
        else:
            print(f"Max CUDA memory allocated was {memtrace.peak} GB")
            print(f"Max CUDA memory reserved was {memtrace.max_reserved} GB")
            print(f"Peak active CUDA memory was {memtrace.peak_active_gb} GB")
            print(f"Cuda Malloc retires : {memtrace.cuda_malloc_retires}")
            print(
                f"CPU Total Peak Memory consumed during the train (max): {memtrace.cpu_peaked + memtrace.cpu_begin} GB"
            )

        # Update the learning rate as needed
        lr_scheduler.step()
        if train_config.enable_fsdp:
            if rank == 0:
                print(
                    f"Epoch {epoch+1}: train_perplexity={train_perplexity:.4f}, train_epoch_loss={train_epoch_loss:.4f}, epcoh time {epoch_end_time}s"
                )
        else:
            print(
                f"Epoch {epoch+1}: train_perplexity={train_perplexity:.4f}, train_epoch_loss={train_epoch_loss:.4f}, epcoh time {epoch_end_time}s"
            )
    avg_epoch_time = sum(epoch_times) / len(epoch_times)
    avg_checkpoint_time = sum(checkpoint_times) / len(checkpoint_times)
    avg_train_prep = sum(train_prep) / len(train_prep)
    avg_train_loss = sum(train_loss) / len(train_loss)

    results["avg_train_prep"] = avg_train_prep
    results["avg_train_loss"] = avg_train_loss
    results["avg_epoch_time"] = avg_epoch_time
    results["avg_checkpoint_time"] = avg_checkpoint_time

    # saving the training params including fsdp setting for reference.
    if train_config.enable_fsdp:
        save_train_params(train_config, fsdp_config, rank)

    return results
