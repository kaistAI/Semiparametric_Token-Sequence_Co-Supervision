from transformers import LlamaForCausalLM, LlamaConfig, LlamaModel
from model.joint_modeling_llama import JointModel

def load_joint_llama_rettoken_from_config(config_path, train_config, tokenizer):
    model_config = LlamaConfig.from_pretrained(config_path) 
    model = JointModel(config=model_config, train_config = train_config)
    model.set_vanilla_vocab_size(len(tokenizer))
    return model

def load_llama_rettoken_from_config(config_path, tokenizer):
    model_config = LlamaConfig.from_pretrained(config_path) 
    model = LlamaModel(config=model_config)
    model.resize_token_embeddings(len(tokenizer))
    return model

def load_llama_causal_rettoken_from_config(config_path, tokenizer):
    model_config = LlamaConfig.from_pretrained(config_path) 
    model = LlamaForCausalLM(config=model_config)
    model.resize_token_embeddings(len(tokenizer))
    model.vanilla_vocab_size = len(tokenizer)
    return model
