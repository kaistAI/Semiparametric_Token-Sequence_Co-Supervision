# Beyond Next Token Prediction: Semiparametric Token Sequence Cosupervision

This is the official codebase of [Beyond Next Token Prediction: Semiparametric Token Sequence Cosupervision](). The repository provides the train and inference code of our main experiment, which is as follows:

1. Train/inference NTP+NSP(Ours)
2. Train/inference NTP(baseline)

This repository is based on the [llama recipes repository](https://github.com/facebookresearch/llama-recipes) from meta. Huge thanks to the contributors! 





# Requirements
## Virtual Environment
To run the examples, make sure to install the requirements using

```
# python 3.9 or higher recommended
pip install -r requirements.txt

```

**Please note that the above requirements.txt will install PyTorch 2.0.1 version, in case you want to run FSDP + PEFT, please make sure to install PyTorch nightlies.**

## GPU Resource
We recommend using 8 A100 80GB NVIDIA GPU Node for training and 1 A6000 NVIDIA GPU Node for inference in order to reproduce our experiments. However, you can follow the most of the fine-tuning strategy used in the [llama recipes repository](https://github.com/facebookresearch/llama-recipes)(We don't support peft method yet.)

## Data
We provide filtered data used to train [Self-RAG](https://arxiv.org/abs/2310.11511).

# 1. Train/inference NTP+NSP(Ours)
## 1.1 Train
We provide training code where both Emb_seq and Gen are initialized from [Llama-2 7B hf ckpt](https://huggingface.co/meta-llama/Llama-2-7b-hf), which is
```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nnodes 1 --master_port=29100 --nproc_per_node 8 run_train.py
```
We also provide the result model of both [Emb_seq](https://huggingface.co/kaist-ai/emb_ntp_nsp_cosupervision_Llama2_7B) and [Gen](https://huggingface.co/kaist-ai/gen_ntp_nsp_cosupervision_Llama2_7B). 
## 1.2. Inference
Modify the --dataset argument to experiment on different dataset.

```
CUDA_VISIBLE_DEVICES=0 accelerate launch run_inference_ntp_nsp.py --dataset kilt_hotpotqa --dist_checkpoint_folder ntp_nsp_cosupervision --ndocs 100
```

# 2. Train/inference NTP(baseline)
As above, Emb_seq and Gen are Llama-2 7B hf ckpt.

## 2.1. Train Emb_seq
```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nnodes 1 --master_port=29100 --nproc_per_node 8 run_train.py --single --dist_checkpoint_folder emb_single_ntp_singlesupervision
```
## 2.2. Train Gen
```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nnodes 1 --master_port=29100 --nproc_per_node 8 run_train_genonly.py --dist_checkpoint_folder gen_ntp_singlesupervision
```
We also provide the result model of both [Emb_seq](https://huggingface.co/kaist-ai/emb_single_ntp_singlesupervision_Llama2_7B) and [Gen](https://huggingface.co/kaist-ai/gen_ntp_singlesupervision_Llama2_7B). Download each model under specific directory name. For 
# 2.3. Inference
Modify the --dataset argument to experiment on different dataset.
```
CUDA_VISIBLE_DEVICES=0 accelerate launch run_inference_ntp.py --dataset kilt_hotpotqa --dist_checkpoint_folder gen_ntp_singlesupervision --ret_checkpoint_folder emb_single_ntp_singlesupervision --ndocs 100 --retriever llama --single
```

