import os
import json
import yaml
from transformers import LlamaTokenizer
from utils.checkpoint_utils import load_sharded_split_single_gpu
from configs import training_config
from model_utils import load_llama_rettoken_from_config, load_joint_llama_rettoken_from_config, load_llama_causal_rettoken_from_config
def check_ckpt_dir_split(model_ckpt_path):
    model_ckpt_files = os.listdir(model_ckpt_path)
    ctx_bin_files, q_bin_files = [], []
    if "ctx" in model_ckpt_files and "question" in model_ckpt_files:
        for file in os.listdir(os.path.join(model_ckpt_path, "ctx")):
            if file.endswith(".bin") or file.endswith(".safetensors"):
                ctx_bin_files.append(file)
        for file in os.listdir(os.path.join(model_ckpt_path, "question")):
            if file.endswith(".bin") or file.endswith(".safetensors"):
                q_bin_files.append(file)       
    if len(ctx_bin_files)>0 and len(q_bin_files)>0:
        return False
    return True

def convert_fsdp_ckpt_to_hf(
    **kwargs
    ):
    with open(kwargs["training_argument"], 'r') as f:
        json_obj = json.load(f)
    train_config = training_config(**json_obj)
    try:
        file_name = 'train_params.yaml'
        # Combine the directory and file name to create the full path
        train_params_path = os.path.join(kwargs["fsdp_checkpoint_path"],file_name)
        # Open the file
        with open(train_params_path, 'r') as file:
            # Load the YAML data
            data = yaml.safe_load(file)

            # Access the 'model_name' field
            HF_model_path_or_name = data.get('model_name')

            print(f"Model name: {HF_model_path_or_name}")
    except FileNotFoundError:
        print(f"The file {train_params_path} does not exist.")
        HF_model_path_or_name = input("Please enter the model name: ")
        print(f"Model name: {HF_model_path_or_name}")
    except Exception as e:
        print(f"An error occurred: {e}")
        
        
    #load the HF model definition from config
    print("model is loaded from config")
    #load the FSDP sharded checkpoints into the model
    tokenizer = LlamaTokenizer.from_pretrained(HF_model_path_or_name)
    tokenizer.add_special_tokens(dict(pad_token="<PAD>"))
    if train_config.natural_form:
        tokenizer.add_special_tokens({'additional_special_tokens': ["[Cs]", "[Ce]"]})
        tokenizer.add_special_tokens({'additional_special_tokens': ["[Ret]"]})
    output_dir = kwargs["consolidated_model_path"]
    ctx_dir = os.path.join(output_dir, "ctx")
    question_dir = os.path.join(output_dir, "question")
    if not os.path.exists(ctx_dir):
        os.makedirs(ctx_dir)
    if not os.path.exists(question_dir):
        os.makedirs(question_dir)
    
    tokenizer.save_pretrained(question_dir)
    tokenizer.save_pretrained(ctx_dir)
    ctx_model,question_model = load_sharded_split_single_gpu(
        load_joint_llama_rettoken_from_config(HF_model_path_or_name, train_config, tokenizer),
        load_llama_rettoken_from_config(HF_model_path_or_name, tokenizer),
        load_llama_causal_rettoken_from_config(HF_model_path_or_name, tokenizer),
        kwargs["fsdp_checkpoint_path"])
    print("model is loaded from FSDP checkpoints")
    #loading the tokenizer form the  model_path

    #save the FSDP sharded checkpoints in HF format
    ctx_model.save_pretrained(ctx_dir)
    question_model.save_pretrained(question_dir)
    print(f"HuggingFace model checkpoints has been saved in {output_dir}")

def delete_distcp_files(model_ckpt_path):
    model_ckpt_files = os.listdir(model_ckpt_path)
    dist_ckpt_files, hf_ckpt_bin_files = [], []
    for file in model_ckpt_files:
        if file.endswith(".distcp") or file.endswith(".metadata") or file.endswith(".yaml"):
            dist_ckpt_files.append(file)
        if file.endswith(".bin") or file.endswith(".safetensors"):
            hf_ckpt_bin_files.append(file)
    assert len(hf_ckpt_bin_files) > 0, "Conversion into HF ckpt files was unsuccessful!"
    for file in dist_ckpt_files:
        if os.path.exists(os.path.join(model_ckpt_path, file)):
            os.remove(os.path.join(model_ckpt_path, file))
            print(f"{file} file deleted!")
        else:
            print(f"{file} does not exist")

def save_to_hf(train_config, model_ckpt_path, kwargs, rank):
    assert os.path.isdir(
        model_ckpt_path
    ), f"{model_ckpt_path} Model Checkpoint Path corresponding to the given config does not exist!"
    if rank == 0:
        is_ckpt_conversion_needed = check_ckpt_dir_split(model_ckpt_path)
        if is_ckpt_conversion_needed:
            convert_fsdp_ckpt_to_hf(
                fsdp_checkpoint_path=model_ckpt_path,
                consolidated_model_path=model_ckpt_path,
                HF_model_path_or_name=train_config.model_name,
                training_argument=kwargs["training_argument"],
            )
        print("Completed converting fsdp ckpt files into hf ckpt files")
        delete_distcp_files(model_ckpt_path)