import os
import sys
from typing import List
import yaml
import time
import json
import fire
import faiss
import torch
import transformers
from datasets import load_dataset
from tqdm import tqdm
import numpy as np
import re
import torch.distributed as dist
from transformers import StoppingCriteriaList, StoppingCriteria

class KeywordsStoppingCriteria(StoppingCriteria):
    def __init__(self, keywords_ids: list, idx):
        self.keywords = keywords_ids
        self.idx = idx

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs
    ) -> bool:
        if input_ids[0][self.idx] in self.keywords:
            return True
        return False

def mean_pooling(token_embeddings, mask):
    token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.0)
    sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
    return sentence_embeddings

def get_embeddings(sentences, tokenizer, model):
    inputs = tokenizer(
        sentences, padding=True, truncation=True, return_tensors="pt"
    ).to(model.device)
    outputs = model(**inputs)
    embeddings = mean_pooling(outputs[0], inputs["attention_mask"])
    return embeddings

def dump(
    model, train_config, ctx_dataloader, local_rank, world_size, do_mean_pooling=False
):
    results = {}
    model.eval()
    ctx_dict = dict()
    for step, batch in enumerate(
        tqdm(ctx_dataloader, colour="green", desc="Embedding Epoch")
    ):
        for key in batch.keys():
            if torch.is_tensor(batch[key]) and local_rank != None:
                if train_config.enable_fsdp:
                    batch[key] = batch[key].to(local_rank)
                else:
                    batch[key] = batch[key].to("cuda:0")
        # Ensure no gradients are computed for this scope to save memory
        with torch.no_grad():
            # Forward pass and compute loss
            if (
                type(model).__name__ == "DistributedDataParallel"
                and "Llama" in model.module.__class__.__name__
            ) or ("Llama" in model.__class__.__name__):
                assert not do_mean_pooling
                outputs = model(
                    input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
                ).last_hidden_state[:, -1, :]
            else:
                if do_mean_pooling:
                    outputs = model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                    )
                    outputs = mean_pooling(outputs[0], batch["attention_mask"])
                else:
                    outputs = model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                    ).last_hidden_state[:, 0, :]
            # print(outputs.shape)
            # outputs = model(**batch)
            labels = batch["target_labels"]
            outputs = outputs.detach().cpu().to(torch.float32).numpy().tolist()
            for label, out in zip(labels, outputs):
                ctx_dict[label] = out
        if train_config.add_vocab:
            for idx, vocab_emb in enumerate(
                model.lm_head.weight.detach().cpu().to(torch.float32).numpy().tolist()
            ):
                ctx_dict[str(idx)] = vocab_emb
    if world_size > 1:
        gathered_ctx_list = [dict() for _ in range(world_size)]
        dist.all_gather_object(gathered_ctx_list, ctx_dict)
        for i in range(1, world_size):
            gathered_ctx_list[0].update(gathered_ctx_list[i])
        gathered_ctx = gathered_ctx_list[0]

    else:
        gathered_ctx = ctx_dict

    return gathered_ctx

def test_ext_dpr_alce(
    model,
    model_dpr,
    train_config,
    dataset_config,
    eval_dataloader,
    gathered_ctx,
    eval_tokenizer,
    ctx_tokenizer,
    local_rank,
    world_size,
):
    results = {}
    ret_idx_dict = {}
    model.eval()
    eval_ids2text = {}

    if dataset_config.eval_docs == "":
        input_path = dataset_config.eval_data
        if input_path.endswith(".json"):
            input_data = json.load(open(input_path))
        else:
            import jsonlines

            with jsonlines.open(input_path, "r") as jsonl_f:
                input_data = [obj for obj in jsonl_f]

        for item in input_data:
            for doc in item["docs"][: dataset_config.ndocs]:
                eval_ids2text[doc["id"]] = f"{doc['title']} :: {doc['text']}"
    else:
        input_path = dataset_config.eval_docs
        input_data = json.load(open(input_path))

        for doc in input_data:
            if "title" in doc.keys():
                eval_ids2text[doc["id"]] = f"{doc['title']} :: {doc['text']}"
            else:
                eval_ids2text[doc["id"]] = doc["text"]

    for step, batch in enumerate(
        tqdm(eval_dataloader, colour="green", desc="Testing Epoch")
    ):
        full_labels = np.array(batch["target_labels"][0])
        full_emb = np.array(
            [gathered_ctx[id] for id in batch["target_labels"][0]], dtype=np.float32
        )
        Index = faiss.IndexFlatIP(full_emb.shape[1])
        Index.add(full_emb)

        with torch.no_grad():
            sequence, ret_idx = _generate_ext_dpr(
                model,
                model_dpr,
                batch["input_ids"],
                eval_tokenizer,
                ctx_tokenizer,
                Index,
                full_labels,
                train_config,
                eval_ids2text,
                max_seq_tokens=eval_tokenizer.model_max_length,
                ret_idx=[],
                is_first=True
            )
            _text = eval_tokenizer.decode(sequence[0])
            print(f"## Output: {_text}")
            results[batch["alce_idx"][0]] = _text
            ret_idx_dict[batch["alce_idx"][0]] = ret_idx
            # re.search(r'## Input:\n\n(.*?)## Output:\n\n', _text).group(1)

    if world_size > 1:
        gathered_results = [dict() for _ in range(world_size)]
        gathered_ret_idx = [dict() for _ in range(world_size)]
        dist.all_gather_object(gathered_results, results)
        dist.all_gather_object(gathered_ret_idx, ret_idx_dict)
        for i in range(1, world_size):
            for k, v in gathered_results[i].items():
                gathered_results[0][k] = v
                gathered_ret_idx[0][k] = v
        gathered_results = gathered_results[0]
        gathered_ret_idx = gathered_ret_idx[0]
    else:
        gathered_results = results
        gathered_ret_idx = ret_idx_dict

    return gathered_results, gathered_ret_idx

def test_ext_contriever_alce(
    model,
    model_dpr,
    train_config,
    dataset_config,
    eval_dataloader,
    gathered_ctx,
    eval_tokenizer,
    ctx_tokenizer,
    local_rank,
    world_size,
):
    results = {}
    ret_idx_dict = {}
    model.eval()
    eval_ids2text = {}

    if dataset_config.eval_docs == "":
        input_path = dataset_config.eval_data
        if input_path.endswith(".json"):
            input_data = json.load(open(input_path))
        else:
            import jsonlines

            with jsonlines.open(input_path, "r") as jsonl_f:
                input_data = [obj for obj in jsonl_f]

        for item in input_data:
            for doc in item["docs"][: dataset_config.ndocs]:
                eval_ids2text[doc["id"]] = f"{doc['title']} :: {doc['text']}"
    else:
        input_path = dataset_config.eval_docs
        input_data = json.load(open(input_path))

        for doc in input_data:
            if "title" in doc.keys():
                eval_ids2text[doc["id"]] = f"{doc['title']} :: {doc['text']}"
            else:
                eval_ids2text[doc["id"]] = doc["text"]

    for step, batch in enumerate(
        tqdm(eval_dataloader, colour="green", desc="Testing Epoch")
    ):
        full_labels = np.array(batch["target_labels"][0])
        full_emb = np.array(
            [gathered_ctx[id] for id in batch["target_labels"][0]], dtype=np.float32
        )
        Index = faiss.IndexFlatIP(full_emb.shape[1])
        Index.add(full_emb)

        with torch.no_grad():
            sequence, ret_idx = _generate_ext_contriever(
                model,
                model_dpr,
                batch["input_ids"],
                eval_tokenizer,
                ctx_tokenizer,
                Index,
                full_labels,
                train_config,
                eval_ids2text,
                max_seq_tokens=eval_tokenizer.model_max_length,
                ret_idx=[],
                is_first=True
            )
            _text = eval_tokenizer.decode(sequence[0])
            print(f"## Output: {_text}")
            results[batch["alce_idx"][0]] = _text
            ret_idx_dict[batch["alce_idx"][0]] = ret_idx
            # re.search(r'## Input:\n\n(.*?)## Output:\n\n', _text).group(1)

    if world_size > 1:
        gathered_results = [dict() for _ in range(world_size)]
        gathered_ret_idx = [dict() for _ in range(world_size)]
        dist.all_gather_object(gathered_results, results)
        dist.all_gather_object(gathered_ret_idx, ret_idx_dict)
        for i in range(1, world_size):
            for k, v in gathered_results[i].items():
                gathered_results[0][k] = v
                gathered_ret_idx[0][k] = v
        gathered_results = gathered_results[0]
        gathered_ret_idx = gathered_ret_idx[0]
    else:
        gathered_results = results
        gathered_ret_idx = ret_idx_dict

    return gathered_results, gathered_ret_idx

def test_alce(
    model,
    train_config,
    dataset_config,
    eval_dataloader,
    gathered_ctx,
    eval_tokenizer,
    local_rank,
    world_size,
    ctx_truncate,
):
    results = {}
    ret_idx_dict = {}
    model.eval()
    eval_ids2text = {}

    if dataset_config.eval_docs == "":
        input_path = dataset_config.eval_data
        if input_path.endswith(".json"):
            input_data = json.load(open(input_path))
        else:
            import jsonlines

            with jsonlines.open(input_path, "r") as jsonl_f:
                input_data = [obj for obj in jsonl_f]

        for item in input_data:
            for doc in item["docs"][: dataset_config.ndocs]:
                eval_ids2text[doc["id"]] = f"{doc['title']} :: {doc['text']}"
    else:
        input_path = dataset_config.eval_docs
        input_data = json.load(open(input_path))

        for doc in input_data:
            if "title" in doc.keys():
                eval_ids2text[doc["id"]] = f"{doc['title']} :: {doc['text']}"
            else:
                eval_ids2text[doc["id"]] = doc["text"]

    for step, batch in enumerate(
        tqdm(eval_dataloader, colour="green", desc="Testing Epoch")
    ):
        full_labels = np.array(batch["target_labels"][0])
        full_emb = np.array(
            [gathered_ctx[id] for id in batch["target_labels"][0]], dtype=np.float32
        )
        Index = faiss.IndexFlatIP(full_emb.shape[1])
        Index.add(full_emb)
        with torch.no_grad():
            sequence, ret_idx = _generate(
                model,
                batch["input_ids"],
                eval_tokenizer,
                Index,
                full_labels,
                train_config,
                eval_ids2text,
                eval_tokenizer.model_max_length,
                [],
                ctx_truncate,
                is_first=True
            )
            _text = eval_tokenizer.decode(sequence[0])
            print(f"*** Output: {_text}")
            results[batch["alce_idx"][0]] = _text
            ret_idx_dict[batch["alce_idx"][0]] = ret_idx
            # re.search(r'## Input:\n\n(.*?)## Output:\n\n', _text).group(1)

    if world_size > 1:
        gathered_results = [dict() for _ in range(world_size)]
        gathered_ret_idx = [dict() for _ in range(world_size)]
        dist.all_gather_object(gathered_results, results)
        dist.all_gather_object(gathered_ret_idx, ret_idx_dict)
        for i in range(1, world_size):
            for k, v in gathered_results[i].items():
                gathered_results[0][k] = v
                gathered_ret_idx[0][k] = v
        gathered_results = gathered_results[0]
        gathered_ret_idx = gathered_ret_idx[0]
    else:
        gathered_results = results
        gathered_ret_idx = ret_idx_dict

    return gathered_results, gathered_ret_idx

def _generate_ext_dpr(
    model,
    model_dpr,
    input_ids,
    tokenizer,
    ctx_tokenizer,
    Index,
    full_labels,
    train_config,
    eval_ids2text,
    max_seq_tokens,
    ret_idx,
    ctx_truncate=False,
    bad_words_ids=None,
    is_first=True
):
    # stop_ids = [tokenizer.convert_ids_to_tokens(id) for id in model.ids2text.keys()]
    # print(stop_ids)
    stop_ids = list(tokenizer.convert_tokens_to_ids(["[Ret]"]))
    cite_start_ids = list(tokenizer.convert_tokens_to_ids(["[Cs]"]))
    # print(stop_ids)
    # print([tokenizer.convert_ids_to_tokens(id) for id in stop_ids])
    cite_stop_ids = list(tokenizer.convert_tokens_to_ids(["[Ce]"]))
    cite_stop_criteria = KeywordsStoppingCriteria(cite_stop_ids, -1)
    stop_criteria = KeywordsStoppingCriteria(stop_ids, -1)
    stop_list = [stop_criteria, cite_stop_criteria]
    if bad_words_ids is None:
        bad_words_ids = [cite_start_ids, cite_stop_ids]
        
    if train_config.ret_first and is_first:
        t_stop_ids = torch.tensor(stop_ids).unsqueeze(0).to(input_ids.device)
        outputs = torch.cat((input_ids, t_stop_ids), dim=1)
    else:
        if type(model).__name__ == "DistributedDataParallel":
            outputs = model.module.generate(
                input_ids=input_ids,
                temperature=0.7,
                top_p=0.9,
                bad_words_ids=bad_words_ids, #[bad_ids],
                do_sample=True,
                num_beams=1,
                stopping_criteria=StoppingCriteriaList(stop_list),
                max_length=max_seq_tokens,
                # use_cache=True,
                # top_k=3,
                # repetition_penalty=1.0,
                # length_penalty=1,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )  
        else:
            outputs = model.generate(
                input_ids=input_ids,
                temperature=0.7,
                top_p=0.9,
                bad_words_ids=bad_words_ids, #[bad_ids],
                do_sample=True,
                num_beams=1,
                stopping_criteria=StoppingCriteriaList(stop_list),
                max_length=max_seq_tokens,
                # use_cache=True,
                # top_k=3,
                # repetition_penalty=1.0,
                # length_penalty=1,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )  
    # print(f"{torch.cuda.current_device()} gen output:", outputs)
    # cnt=0
    # for item in outputs.hidden_states:#(batch_size, generated_length, hidden_size)
    #     hidden_state = item[-1][:, -1, :]#1, 1, 4096
    #     logit = model.lm_head(hidden_state)#4096, 32004
    #     print(cnt, torch.topk(logit, 100, -1), logit[:,21776], logit[:,32001], logit[:,32002], logit[:,32003])
    #     cnt+=1
    # print("hidden output:", len(outputs.hidden_states), len(outputs.hidden_states[-1]), outputs.hidden_states[-1][-1].shape)

    # print("temp result:", tokenizer.convert_ids_to_tokens(outputs.sequences[0]))
    # new_tokens = outputs.sequences.shape[1]-input_ids.shape[1]
    if outputs[0][-1] in stop_ids and max_seq_tokens > outputs.shape[1]:
        question_txt = tokenizer.decode(outputs[0][:-1], skip_special_tokens=True)
        question_encoded = ctx_tokenizer(
            question_txt, padding=True, truncation=True, return_tensors="pt"
        )
        for key in question_encoded.keys():
            if torch.is_tensor(question_encoded[key]):
                question_encoded[key] = question_encoded[key].to(model_dpr.device)
        if (
            type(model_dpr).__name__ == "DistributedDataParallel"
            and "Llama" in model_dpr.module.__class__.__name__
        ) or ("Llama" in model_dpr.__class__.__name__):
            hidden_state = model_dpr(input_ids = outputs)[0][:, -1]
        else:
            hidden_state = model_dpr(**question_encoded)[0][:, 0]
        # print("hidden:", hidden_state.shape)
        distances, indices = Index.search(
            np.array(hidden_state.detach().cpu().to(torch.float32), dtype=np.float32), 1
        )
        ret_idx.append(full_labels[indices[0][0]])

        temp_idx = []
        for i, elem in enumerate(outputs[0][:-1]):
            if elem == cite_start_ids[0]:
                temp_idx.append(i)
        prev_seq = None
        # print(f"[1] temp_idx: {temp_idx}")
        if len(temp_idx) > 0:
            temp_output = outputs[0][temp_idx[-1] : -2]
            if cite_start_ids[0] in temp_output and cite_stop_ids[0] not in temp_output:
                prev_seq = torch.cat(
                    [
                        outputs[:, :-1],
                        torch.Tensor([tokenizer.convert_tokens_to_ids(["[Ce]"])]).to(
                            model.device, dtype=outputs.dtype
                        ),
                        torch.Tensor([tokenizer.convert_tokens_to_ids(["[Ret]"])]).to(
                            model.device, dtype=outputs.dtype
                        ),
                    ],
                    dim=-1,
                )
        if prev_seq is None:
            prev_seq = outputs

        # if tokenizer.convert_tokens_to_ids("[Cs]") in outputs and tokenizer.convert_tokens_to_ids("[Ce]") not in outputs:
        #     prev_seq = torch.cat([
        #         outputs[:, :-1],
        #         torch.Tensor([tokenizer.convert_tokens_to_ids(["[Ce]"])]).to(model.device, dtype = outputs.sequences.dtype),
        #         torch.Tensor([tokenizer.convert_tokens_to_ids(["[RET]"])]).to(model.device, dtype = outputs.sequences.dtype)
        #     ], dim=-1)
        # else:
        #     prev_seq = outputs

        if max_seq_tokens > outputs.shape[1]:
            outputs = torch.cat(
                [
                    prev_seq,
                    tokenizer(
                        eval_ids2text[full_labels[indices[0][0]]],
                        return_tensors="pt",
                        add_special_tokens=False,
                    ).input_ids.to(model.device, dtype=outputs.dtype),
                    torch.Tensor([tokenizer.convert_tokens_to_ids(["[Cs]"])]).to(
                        model.device, dtype=outputs.dtype
                    ),
                ],
                dim=-1,
            )

            return _generate_ext_dpr(
                model,
                model_dpr,
                outputs,
                tokenizer,
                ctx_tokenizer,
                Index,
                full_labels,
                train_config,
                eval_ids2text,
                max_seq_tokens,
                ret_idx,
                bad_words_ids=[cite_start_ids],
                is_first=False
            )
        else:
            outputs = torch.cat(
                [
                    prev_seq,
                    tokenizer(
                        eval_ids2text[full_labels[indices[0][0]]],
                        return_tensors="pt",
                        add_special_tokens=False,
                    ).input_ids.to(model.device, dtype=outputs.dtype),
                    torch.Tensor([tokenizer.convert_tokens_to_ids(["[Cs]"])]).to(
                        model.device, dtype=outputs.dtype
                    ),
                    torch.Tensor([tokenizer.convert_tokens_to_ids(["[Ce]"])]).to(
                        model.device, dtype=outputs.dtype
                    ),
                ],
                dim=-1,
            )
            return outputs, ret_idx

    elif outputs[0][-1] in cite_stop_ids and max_seq_tokens > outputs.shape[1]:
        # [Ce] 때문에 멈춘 경우
        if ctx_truncate:
            # print("before trunc: ", outputs.sequences[0])
            trunc_start_list = torch.where(outputs == stop_ids[0])[
                -1
            ].tolist()  #  find [RET]
            if len(trunc_start_list) == 0:
                return outputs, ret_idx
            trunc_start = trunc_start_list[-1]
            trunc_end_list = torch.where(outputs == cite_start_ids[0])[
                -1
            ].tolist()  # find [Cs]
            if len(trunc_end_list) == 0:
                return outputs, ret_idx
            trunc_end = trunc_end_list[-1]
            stitched_outputs = torch.cat(
                [outputs[:, :trunc_start], outputs[:, trunc_end:]], dim=-1
            )
            assert (
                cite_stop_ids[0] in stitched_outputs
            )  # assert [Ce] in stitched_outputs

            if max_seq_tokens > outputs.shape[1]:
                return _generate_ext_dpr(model, model_dpr, stitched_outputs, tokenizer,ctx_tokenizer, Index, full_labels, train_config,eval_ids2text, max_seq_tokens, ret_idx, ctx_truncate, bad_words_ids=[[cite_start_ids[0]], [cite_stop_ids[0]]], is_first=False)
                # return _generate(model, stitched_outputs, tokenizer, Index, full_labels, train_config,eval_ids2text, max_seq_tokens, ret_idx, ctx_truncate, bad_words_ids=[[cite_start_ids[0]], [cite_stop_ids[0]]])
            else:
                return stitched_outputs, ret_idx
        else:
            if max_seq_tokens > outputs.shape[1]:
                return _generate_ext_dpr(model, model_dpr, outputs, tokenizer,ctx_tokenizer, Index, full_labels, train_config,eval_ids2text, max_seq_tokens, ret_idx, ctx_truncate, bad_words_ids=[[cite_start_ids[0]], [cite_stop_ids[0]]], is_first=False)
                # return _generate(model, outputs, tokenizer, Index, full_labels, train_config,eval_ids2text, max_seq_tokens, ret_idx, ctx_truncate, bad_words_ids=[[cite_start_ids[0]], [cite_stop_ids[0]]])
            else:
                return outputs, ret_idx

    else:
        temp_idx = []
        for i, elem in enumerate(outputs[0]):
            if elem == cite_start_ids[0]:
                temp_idx.append(i)
        if len(temp_idx) > 0 and cite_stop_ids[0] not in outputs[0][temp_idx[-1] :]:
            outputs = torch.cat(
                [
                    outputs,
                    torch.Tensor([tokenizer.convert_tokens_to_ids(["[Ce]"])]).to(
                        model.device, dtype=outputs.dtype
                    ),
                ],
                dim=-1,
            )
        return outputs, ret_idx


def _generate_ext_contriever(model, model_contriever, input_ids, tokenizer,ctx_tokenizer, Index, full_labels, train_config,  eval_ids2text, max_seq_tokens, ret_idx, ctx_truncate=False, bad_words_ids=None, is_first=True):
    stop_ids = list(tokenizer.convert_tokens_to_ids(["[Ret]"]))
    cite_start_ids = list(tokenizer.convert_tokens_to_ids(["[Cs]"]))
    # print(stop_ids)
    # print([tokenizer.convert_ids_to_tokens(id) for id in stop_ids])
    cite_stop_ids = list(tokenizer.convert_tokens_to_ids(["[Ce]"]))
    cite_stop_criteria = KeywordsStoppingCriteria(cite_stop_ids, -1)
    stop_criteria = KeywordsStoppingCriteria(stop_ids, -1)
    stop_list = [stop_criteria, cite_stop_criteria]
    if bad_words_ids is None:
        bad_words_ids = [cite_start_ids, cite_stop_ids]

    if train_config.ret_first and is_first:
        t_stop_ids = torch.tensor(stop_ids).unsqueeze(0).to(input_ids.device)
        outputs = torch.cat((input_ids, t_stop_ids), dim=1)
    else:
        if type(model).__name__ == "DistributedDataParallel":
            outputs = model.module.generate(
                input_ids=input_ids,
                temperature=0.7,
                top_p=0.9,
                bad_words_ids=bad_words_ids,
                do_sample=True,
                num_beams=1,
                stopping_criteria=StoppingCriteriaList(stop_list),
                max_length=max_seq_tokens,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )  
        else:
            outputs = model.generate(
                input_ids=input_ids,
                temperature=0.7,
                top_p=0.9,
                bad_words_ids=bad_words_ids,
                do_sample=True,
                num_beams=1,
                stopping_criteria=StoppingCriteriaList(stop_list),
                max_length=max_seq_tokens,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )  

    if outputs[0][-1] in stop_ids and max_seq_tokens > outputs.shape[1]:
        question_txt = tokenizer.decode(outputs[0][:-1], skip_special_tokens=True)
        # question_encoded = ctx_tokenizer(question_txt, padding=True, truncation=True, return_tensors='pt')
        # for key in question_encoded.keys():
        #     if torch.is_tensor(question_encoded[key]):
        #         question_encoded[key] = question_encoded[key].to(model_contriever.device)
        if (
            type(model_contriever).__name__ == "DistributedDataParallel"
            and "Llama" in model_contriever.module.__class__.__name__
        ) or ("Llama" in model_contriever.__class__.__name__):
            assert False
            hidden_state = model_contriever(**question_encoded)[0][:, -1]
        else:
            hidden_state = get_embeddings(question_txt, ctx_tokenizer, model_contriever)
        distances, indices = Index.search(
            np.array(hidden_state.detach().cpu().to(torch.float32), dtype=np.float32), 1
        )
        ret_idx.append(full_labels[indices[0][0]])

        temp_idx = []
        for i, elem in enumerate(outputs[0][:-1]):
            if elem == cite_start_ids[0]:
                temp_idx.append(i)
        prev_seq = None
        if len(temp_idx) > 0:
            temp_output = outputs[0][temp_idx[-1] : -2]
            if cite_start_ids[0] in temp_output and cite_stop_ids[0] not in temp_output:
                prev_seq = torch.cat(
                    [
                        outputs[:, :-1],
                        torch.Tensor([tokenizer.convert_tokens_to_ids(["[Ce]"])]).to(
                            model.device, dtype=outputs.dtype
                        ),
                        torch.Tensor([tokenizer.convert_tokens_to_ids(["[Ret]"])]).to(
                            model.device, dtype=outputs.dtype
                        ),
                    ],
                    dim=-1,
                )
        if prev_seq is None:
            prev_seq = outputs

        if max_seq_tokens > outputs.shape[1]:
            outputs = torch.cat(
                [
                    prev_seq,
                    tokenizer(
                        eval_ids2text[full_labels[indices[0][0]]],
                        return_tensors="pt",
                        add_special_tokens=False,
                    ).input_ids.to(model.device, dtype=outputs.dtype),
                    torch.Tensor([tokenizer.convert_tokens_to_ids(["[Cs]"])]).to(
                        model.device, dtype=outputs.dtype
                    ),
                ],
                dim=-1,
            )

            return _generate_ext_contriever(model, model_contriever, outputs, tokenizer,ctx_tokenizer, Index, full_labels, train_config,eval_ids2text, max_seq_tokens, ret_idx, bad_words_ids = [cite_start_ids], is_first=False)
        else:
            outputs = torch.cat(
                [
                    outputs,
                    tokenizer(
                        eval_ids2text[full_labels[indices[0][0]]],
                        return_tensors="pt",
                        add_special_tokens=False,
                    ).input_ids.to(model.device, dtype=outputs.dtype),
                    torch.Tensor([tokenizer.convert_tokens_to_ids(["[Cs]"])]).to(
                        model.device, dtype=outputs.dtype
                    ),
                    torch.Tensor([tokenizer.convert_tokens_to_ids(["[Ce]"])]).to(
                        model.device, dtype=outputs.dtype
                    ),
                ],
                dim=-1,
            )
            return outputs, ret_idx

    elif outputs[0][-1] in cite_stop_ids and max_seq_tokens > outputs.shape[1]:
        # [Ce] 때문에 멈춘 경우
        if ctx_truncate:
            # print("before trunc: ", outputs.sequences[0])
            trunc_start_list = torch.where(outputs == stop_ids[0])[
                -1
            ].tolist()  #  find [RET]
            if len(trunc_start_list) == 0:
                return outputs, ret_idx
            trunc_start = trunc_start_list[-1]
            trunc_end_list = torch.where(outputs == cite_start_ids[0])[
                -1
            ].tolist()  # find [Cs]
            if len(trunc_end_list) == 0:
                return outputs, ret_idx
            trunc_end = trunc_end_list[-1]
            stitched_outputs = torch.cat(
                [outputs[:, :trunc_start], outputs[:, trunc_end:]], dim=-1
            )
            assert (
                cite_stop_ids[0] in stitched_outputs
            )  # assert [Ce] in stitched_outputs

            if max_seq_tokens > outputs.shape[1]:
                return _generate_ext_contriever(model, model_contriever, stitched_outputs, tokenizer,ctx_tokenizer, Index, full_labels, train_config,eval_ids2text, max_seq_tokens, ret_idx, bad_words_ids=[[cite_start_ids[0]], [cite_stop_ids[0]]], is_first=False)
            else:
                return stitched_outputs, ret_idx
        else:
            if max_seq_tokens > outputs.shape[1]:
                # return _generate(model, outputs, tokenizer, Index, full_labels, train_config,eval_ids2text, max_seq_tokens, ret_idx, ctx_truncate, bad_words_ids=[[cite_start_ids[0]], [cite_stop_ids[0]]])
                return _generate_ext_contriever(model, model_contriever, outputs, tokenizer,ctx_tokenizer, Index, full_labels, train_config,eval_ids2text, max_seq_tokens, ret_idx, bad_words_ids=[[cite_start_ids[0]], [cite_stop_ids[0]]], is_first=False)
            else:
                return outputs, ret_idx

    else:
        temp_idx = []
        for i, elem in enumerate(outputs[0]):
            if elem == cite_start_ids[0]:
                temp_idx.append(i)
        # print(f"[3] temp_idx: {temp_idx}")
        if len(temp_idx) > 0 and cite_stop_ids[0] not in outputs[0][temp_idx[-1] :]:
            outputs = torch.cat(
                [
                    outputs,
                    torch.Tensor([tokenizer.convert_tokens_to_ids(["[Ce]"])]).to(
                        model.device, dtype=outputs.dtype
                    ),
                ],
                dim=-1,
            )
        return outputs, ret_idx

def _generate(
    model,
    input_ids,
    tokenizer,
    Index,
    full_labels,
    train_config,
    eval_ids2text,
    max_seq_tokens,
    ret_idx,
    ctx_truncate=False,
    bad_words_ids=None,
    is_first=True
):
    stop_ids = list(tokenizer.convert_tokens_to_ids(["[Ret]"]))
    stop_criteria = KeywordsStoppingCriteria(stop_ids, -2)
    cite_start_ids = list(tokenizer.convert_tokens_to_ids(["[Cs]"]))
    cite_stop_ids = list(tokenizer.convert_tokens_to_ids(["[Ce]"]))
    cite_stop_criteria = KeywordsStoppingCriteria(cite_stop_ids, -1)
    #    if ctx_truncate:
    stop_list = [stop_criteria, cite_stop_criteria]
    if bad_words_ids is None:
        bad_words_ids = [cite_start_ids]
        # bad_words_ids = [cite_start_ids, cite_stop_ids]
    # else:
    # stop_list = [stop_criteria]
    # print(input_ids.shape)
    # print("temp result before generate:", tokenizer.decode(input_ids[0]))

    if train_config.ret_first and is_first:
        t_stop_ids = torch.tensor(stop_ids).unsqueeze(0).to(input_ids.device)
        input_ids = torch.cat((input_ids, t_stop_ids), dim=1)
        # assert stop_ids not in input_ids
        # outputs = torch.cat((input_ids, t_stop_ids, t_stop_ids), dim=1)
    
    if type(model).__name__ == "DistributedDataParallel":
        outputs = model.module.generate(
            input_ids=input_ids,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            num_beams=1,
            bad_words_ids=bad_words_ids,  # [cite_start_ids],
            stopping_criteria=StoppingCriteriaList(stop_list),
            max_length=max_seq_tokens,
            # use_cache=True,
            # top_k=3,
            # repetition_penalty=1.0,
            # length_penalty=1,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            return_dict_in_generate=True,
            output_hidden_states=True,
        )
    else:
        outputs = model.generate(
            input_ids=input_ids,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            num_beams=1,
            bad_words_ids=bad_words_ids,  # [cite_start_ids],
            stopping_criteria=StoppingCriteriaList(stop_list),
            max_length=max_seq_tokens,
            # use_cache=True,
            # top_k=3,
            # repetition_penalty=1.0,
            # length_penalty=1,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            return_dict_in_generate=True,
            output_hidden_states=True,
        )

    # print(f"output: {tokenizer.decode(outputs.sequences[0])}")

    seq = outputs.sequences
    # print(f"[output] {tokenizer.decode(seq[0])}")
    # input()

    ### if ctxtruncate -> add [Ce] if it stops by [Ret]


    if seq[0][-2] in stop_ids and max_seq_tokens>seq.shape[1]-1:
        # RET token 생성해서 멈춘 경우
        # print(f"\n\n** Stop because of [Ret]")
        distances, indices = Index.search(
            np.array(
                outputs.hidden_states[-1][-1][:, -1, :]
                .detach()
                .cpu()
                .to(torch.float32),
                dtype=np.float32,
            ),
            1,
        )

        ret_idx.append(full_labels[indices[0][0]])
        # print(f"[1] Append {full_labels[indices[0][0]]}: {eval_ids2text[full_labels[indices[0][0]]]} in ret_idx")

        ## add [Ce] in generated response if it doesn't contain one while it does contain [Cs]
        temp_idx = []
        for i, elem in enumerate(seq[0][:-2]):
            if elem == cite_start_ids[0]:
                temp_idx.append(i)
        prev_seq = None
        if len(temp_idx) > 0:
            temp_output = seq[0][temp_idx[-1]:-2]
            if cite_start_ids[0] in temp_output and cite_stop_ids[0] not in temp_output:
                # print(f"[1] temp_output: {temp_output} || Cs: {cite_start_ids[0]} || Ce: {cite_stop_ids[0]}")
                prev_seq = torch.cat([
                    seq[:, :-2],
                    torch.Tensor([tokenizer.convert_tokens_to_ids(["[Ce]"])]).to(model.device, dtype = seq.dtype),
                    torch.Tensor([tokenizer.convert_tokens_to_ids(["[Ret]"])]).to(model.device, dtype = seq.dtype)
                ], dim=-1)
        if prev_seq is None:
            # print(f"[1] prev_seq is None!")
            prev_seq = seq[:, :-1]
        # print(f"[1] prev_seq: {tokenizer.decode(prev_seq[0])}")

        stitched_outputs = torch.cat(
            [prev_seq, 
            tokenizer(eval_ids2text[full_labels[indices[0][0]]], return_tensors="pt", add_special_tokens=False).input_ids.to(model.device, dtype = seq.dtype), 
            torch.Tensor([tokenizer.convert_tokens_to_ids(["[Cs]"])]).to(model.device, dtype = seq.dtype)
            ], dim=-1)
        # print(f"[1] stitched_output: {tokenizer.decode(stitched_outputs[0])}")

        if max_seq_tokens > stitched_outputs.shape[1]:
            if ctx_truncate:
                return _generate(model, stitched_outputs, tokenizer, Index, full_labels, train_config,eval_ids2text, max_seq_tokens, ret_idx, ctx_truncate, bad_words_ids=[[cite_start_ids[0]], [stop_ids[0]]], is_first=False)
            else:
                return _generate(
                    model,
                    stitched_outputs,
                    tokenizer,
                    Index,
                    full_labels,
                    train_config,
                    eval_ids2text,
                    max_seq_tokens,
                    ret_idx,
                    ctx_truncate,
                    is_first=False
                )
        else:
            # stitched_outputs = torch.cat(
            #     [prev_seq, 
            #     tokenizer(eval_ids2text[full_labels[indices[0][0]]], return_tensors="pt", add_special_tokens=False).input_ids.to(model.device, dtype = seq.dtype), 
            #     torch.Tensor([tokenizer.convert_tokens_to_ids(["[Cs]"])]).to(model.device, dtype = seq.dtype),
            #     torch.Tensor([tokenizer.convert_tokens_to_ids(["[Ce]"])]).to(model.device, dtype = seq.dtype)
            # ], dim=-1)
            return prev_seq, ret_idx
    
    elif seq[0][-1] in cite_stop_ids and max_seq_tokens>seq.shape[1]-1:
        # print(f"\n\n** Stop because of [Ce]")
        # [Ce] 때문에 멈춘 경우
        if ctx_truncate:
            # print(f"[2] ctx_truncate")
            # assert False #  temporal
            # print("** before trunc: ", tokenizer.decode(seq[0]))
            trunc_start_list = torch.where(seq==stop_ids[0])[-1].tolist() #  find [RET]
            if len(trunc_start_list)==0:
                # print("** trunc_start_list == 0")
                return seq, ret_idx
            trunc_start = trunc_start_list[-1]
            trunc_end_list = torch.where(seq==cite_start_ids[0])[-1].tolist()  # find [Cs]
            if len(trunc_end_list)==0:
                # print("** trunc_end_list== 0")
                return seq, ret_idx
            trunc_end = trunc_end_list[-1]
            stitched_outputs = torch.cat(
                [seq[:, :trunc_start], 
                seq[:, trunc_end:]
                ], dim=-1)
            # print(f"** trunc_start: {trunc_start} || trunc_end: {trunc_end}")
            # print("after trunc: ", tokenizer.decode(stitched_outputs[0]))
            assert cite_stop_ids[0] in stitched_outputs # assert [Ce] in stitched_outputs
            assert stop_ids[0] not in stitched_outputs # assert [Ce] in stitched_outputs
            if max_seq_tokens > seq.shape[1]:
                return _generate(model, stitched_outputs, tokenizer, Index, full_labels, train_config,eval_ids2text, max_seq_tokens, ret_idx, ctx_truncate, bad_words_ids=[[cite_start_ids[0]], [cite_stop_ids[0]]], is_first=False)
            else:
                return stitched_outputs, ret_idx
        else:
            # print(f"[2] NOT ctx_truncate")
            if max_seq_tokens > seq.shape[1]:
                # print(f"[2] output sequence: {tokenizer.decode(seq[0])}")
                try:
                    return _generate(model, seq, tokenizer, Index, full_labels, train_config,eval_ids2text, max_seq_tokens, ret_idx, ctx_truncate, bad_words_ids=[[cite_start_ids[0]], [cite_stop_ids[0]]], is_first=False)
                except:
                    return seq, ret_idx
            else:
                return seq, ret_idx
    else:
        # print(f"** Stop in else!")
        temp_idx = []
        for i, elem in enumerate(seq[0]):
            if elem == cite_start_ids[0]:
                temp_idx.append(i)
        # print(f"** ")
        if len(temp_idx) > 0 and cite_stop_ids[0] not in seq[0][temp_idx[-1]:]:
            # print(f"add [Ce]")
            seq = torch.cat([
                seq, 
                torch.Tensor([tokenizer.convert_tokens_to_ids(["[Ce]"])]).to(model.device, dtype=seq.dtype)
                ], dim=-1)   
        # print(f"[3] output sequence: {tokenizer.decode(seq[0])}")

        return seq, ret_idx