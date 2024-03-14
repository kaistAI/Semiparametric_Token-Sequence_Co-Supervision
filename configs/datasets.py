# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.
from dataclasses import dataclass
from typing import ClassVar

@dataclass
class selfrag_multi_dataset:
    dataset: str = "selfrag_multi_dataset"
    train_data_path: str = (
        "dataset/finetuning_dataset/selfrag_multi/natural_np_free.train.pickle"
    )
    train_ids2text: str = "dataset/finetuning_dataset/selfrag_multi/ids2text.json"
    loss_mask_context: str = "no_mask"
    train_sample: int = -1
    allow_noret: bool = False
    put_hardneg: bool = False


@dataclass
class kilt_fever:
    dataset: str = "kilt_fever"
    eval_data: str = "dataset/eval/kilt-fever/done.fever.contriever_msmarco.top100.json"
    eval_docs: str = ""
    ndocs: int = 100
    ctx_truncate: bool = False
    task: str = "fever"
    max_new_tokens: int = 300


@dataclass
class kilt_wow:
    dataset: str = "kilt_wow"
    eval_data: str = "dataset/eval/kilt-wow/done.wow.contriever_msmarco.top100.json"
    eval_docs: str = ""
    ndocs: int = 100
    ctx_truncate: bool = False
    task: str = "wow"
    max_new_tokens: int = 300

@dataclass
class kilt_zsre:
    dataset: str = "kilt_zsre"
    eval_data: str = "dataset/eval/kilt-zsre/n_done.zsre.contriever_msmarco.top100.json"
    eval_docs: str = ""
    ndocs: int = 100
    ctx_truncate: bool = False
    task: str = "zsre"
    max_new_tokens: int = 300


@dataclass
class kilt_trex:
    dataset: str = "kilt_trex"
    eval_data: str = "dataset/eval/kilt-trex/n_done.trex.contriever_msmarco.top100.json"
    eval_docs: str = ""
    ndocs: int = 100
    ctx_truncate: bool = False
    task: str = "trex"
    max_new_tokens: int = 300

@dataclass
class kilt_zsre:
    dataset: str = "kilt_zsre"
    eval_data: str = "dataset/eval/kilt-zsre/n_done.zsre.contriever_msmarco.top100.json"
    eval_docs: str = ""
    ndocs: int = 100
    ctx_truncate: bool = False
    task: str = "zsre"
    max_new_tokens: int = 300


@dataclass
class kilt_trex:
    dataset: str = "kilt_trex"
    eval_data: str = "dataset/eval/kilt-trex/n_done.trex.contriever_msmarco.top100.json"
    eval_docs: str = ""
    ndocs: int = 100
    ctx_truncate: bool = False
    task: str = "trex"
    max_new_tokens: int = 300

@dataclass
class kilt_eli5:
    dataset: str = "kilt_eli5"
    eval_data: str = "dataset/eval/kilt-eli5/n_done.eli5.contriever_msmarco.top100.json"
    eval_docs: str = ""
    ndocs: int = 100
    ctx_truncate: bool = False
    task: str = "eli5"
    max_new_tokens: int = 300


@dataclass
class kilt_triviaqa:
    dataset: str = "kilt_triviaqa"
    eval_data: str = "dataset/eval/kilt-triviaqa/n_done.triviaqa.contriever_msmarco.top100.json"
    eval_docs: str = ""
    ndocs: int = 100
    ctx_truncate: bool = False
    task: str = "triviaqa"
    max_new_tokens: int = 300

@dataclass
class kilt_nq:
    dataset: str = "kilt_nq"
    eval_data: str= "dataset/eval/kilt-nq/done.nq.contriever_msmarco.top100.json"
    eval_docs: str= ""
    ndocs: int = 100
    ctx_truncate: bool = False
    task: str = "nq"
    max_new_tokens: int = 300
