# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from dataclasses import dataclass


@dataclass
class dpr_nq_dataset:
    dataset: str = "dpr_nq_dataset"
    train_data_path: str = "dataset/dpr/nq/train.full.pickle"
    train_ids2text: str = "dataset/dpr/nq/train.ids2text.json"
    input_query: bool = True
    data_type: str = "none"
    loss_mask_context: str = "no_mask"
    eval_data_path: str = "dataset/dpr/nq/dev.full.pickle"
    eval_ids2text: str = "dataset/dpr/nq/dev.ids2text.json"
    wiki_data_path: str = None
    wiki_data_sample_ratio: float = None
    add_instruction: bool = True
    put_hardneg: bool = False
    train_sample: int = -1
    val_sample: int = -1
    allow_noret: bool = False
    rag_ctx_q: bool = False
    rag_q_ctx: bool = False


@dataclass
class dpr_nq_hardneg_dataset:
    dataset: str = "dpr_nq_hardneg_dataset"
    train_data_path: str = "dataset/dpr/nq/train.full.withHardNeg.pickle"
    train_ids2text: str = "dataset/dpr/nq/train.ids2text.json"
    input_query: bool = True
    data_type: str = "none"
    loss_mask_context: str = "no_mask"
    eval_data_path: str = "dataset/dpr/nq/dev.full.pickle"
    eval_ids2text: str = "dataset/dpr/nq/dev.ids2text.json"
    wiki_data_path: str = None
    wiki_data_sample_ratio: float = None
    add_instruction: bool = True
    put_hardneg: bool = True
    train_sample: int = -1
    val_sample: int = -1
    allow_noret: bool = False
    rag_ctx_q: bool = False
    rag_q_ctx: bool = False


@dataclass
class dpr_kilt_dataset:
    dataset: str = "dpr_kilt_dataset"
    train_data_path: str = (
        "dataset/kilt/qa_dataset/nq/natural_np_free.train15000.pickle"
    )
    train_ids2text: str = "dataset/kilt/qa_dataset/nq/ids2text.json"
    input_query: bool = True
    data_type: str = "none"
    loss_mask_context: str = "context"
    eval_data_path: str = "dataset/kilt/qa_dataset/nq/natural_np_free.test50.pickle"
    eval_ids2text: str = "dataset/kilt/qa_dataset/nq/test50_ids2text.json"
    add_instruction: bool = True
    put_hardneg: bool = False
    train_sample: int = -1
    val_sample: int = -1
    allow_noret: bool = False
    rag_ctx_q: bool = False
    rag_q_ctx: bool = False


@dataclass
class dpr_kilt_dataset_augmented:
    dataset: str = "dpr_kilt_dataset_augmented"
    train_data_path: str = (
        "dataset/kilt/qa_dataset/nq_augmented/natural_np_free.train.pickle"
    )
    train_ids2text: str = "dataset/kilt/qa_dataset/nq_augmented/ids2text.json"
    input_query: bool = True
    data_type: str = "none"
    loss_mask_context: str = "no_mask"
    eval_data_path: str = (
        "dataset/kilt/qa_dataset/nq_augmented/natural_np_free.test50.pickle"
    )
    eval_ids2text: str = "dataset/kilt/qa_dataset/nq_augmented/test50_ids2text.json"
    add_instruction: bool = True
    put_hardneg: bool = False
    train_sample: int = -1
    val_sample: int = -1
    allow_noret: bool = False
    rag_ctx_q: bool = False
    rag_q_ctx: bool = False


@dataclass
class joint_kilt_dataset:
    dataset: str = "joint_kilt_dataset"
    train_data_path: str = (
        "dataset/kilt/qa_dataset/nq/natural_np_free.train15000.pickle"
    )
    train_ids2text: str = "dataset/kilt/qa_dataset/nq/ids2text.json"
    input_query: bool = True
    train_data_type: str = "pretrain"
    eval_data_type: str = "pretrain"
    loss_mask_context: str = "context"
    eval_data_path: str = "dataset/kilt/qa_dataset/nq/natural_np_free.test50.pickle"
    eval_ids2text: str = "dataset/kilt/qa_dataset/nq/test50_ids2text.json"
    wiki_data_path: str = None
    wiki_data_sample_ratio: float = None
    add_instruction: bool = True
    put_hardneg: bool = False
    train_sample: int = -1
    val_sample: int = -1
    allow_noret: bool = False
    rag_ctx_q: bool = False
    rag_q_ctx: bool = False


@dataclass
class dpr_nq_wiki_dataset:
    dataset: str = "dpr_wiki_dataset"
    train_data_path: str = "dataset/dpr/nq/train.full.pickle"
    train_ids2text: str = "dataset/dpr/nq/train.ids2text.json"
    input_query: bool = True
    data_type: str = "none"
    loss_mask_context: str = "no_mask"
    eval_data_path: str = "dataset/dpr/nq/dev.full.pickle"
    eval_ids2text: str = "dataset/dpr/nq/dev.ids2text.json"
    wiki_data_path: str = "/home/jihoon/downloads/data/wikipedia_split/psgs_w100.tsv"
    wiki_data_sample_ratio: float = 0.1
    add_instruction: bool = True
    put_hardneg: bool = False
    oracle: bool = False
    do_swap: bool = False
    train_sample: int = -1
    val_sample: int = -1
    allow_noret: bool = False
    rag_ctx_q: bool = False
    rag_q_ctx: bool = False


@dataclass
class joint_kilt_dataset_revised:
    dataset: str = "joint_kilt_dataset_revised"
    train_data_path: str = (
        "dataset/kilt/qa_dataset/nq_revised/natural_np_free.train.pickle"
    )
    train_ids2text: str = "dataset/kilt/qa_dataset/nq_revised/ids2text.json"
    input_query: bool = True
    train_data_type: str = "pretrain"
    eval_data_type: str = "pretrain"
    loss_mask_context: str = "context"
    eval_data_path: str = (
        "dataset/kilt/qa_dataset/nq_revised/natural_np_free.test50.pickle"
    )
    eval_ids2text: str = "dataset/kilt/qa_dataset/nq_revised/test50_ids2text.json"
    add_instruction: bool = True
    put_hardneg: bool = False
    oracle: bool = False
    do_swap: bool = False
    train_sample: int = -1
    val_sample: int = -1
    allow_noret: bool = False
    rag_ctx_q: bool = False
    rag_q_ctx: bool = False


@dataclass
class joint_kilt_dataset_rettoken_augmented:
    dataset: str = "joint_kilt_dataset_augmented"
    train_data_path: str = (
        "dataset/kilt/qa_dataset/nq_augmented/natural_np_free.train.pickle"
    )
    train_ids2text: str = "dataset/kilt/qa_dataset/nq_augmented/ids2text.json"
    input_query: bool = True
    train_data_type: str = "pretrain"
    eval_data_type: str = "pretrain"
    loss_mask_context: str = "no_mask"
    eval_data_path: str = (
        "dataset/kilt/qa_dataset/nq_augmented/natural_np_free.test.pickle"
    )
    eval_ids2text: str = "dataset/kilt/qa_dataset/nq_augmented/test_ids2text.json"
    add_instruction: bool = True
    put_hardneg: bool = False
    oracle: bool = False
    do_swap: bool = False
    train_sample: int = -1
    val_sample: int = -1
    allow_noret: bool = False
    rag_ctx_q: bool = False
    rag_q_ctx: bool = False


@dataclass
class joint_kilt_dataset_rettoken_augmented_synthetic_instruction:
    dataset: str = "joint_kilt_dataset_augmented"
    train_data_path: str = (
        "dataset/kilt/qa_dataset/nq_augmented/natural_np_free.train.pickle"
    )
    train_ids2text: str = "dataset/kilt/qa_dataset/nq_augmented/ids2text.json"
    input_query: bool = True
    train_data_type: str = "synthetic_instruction"
    eval_data_type: str = "synthetic_instruction"
    loss_mask_context: str = "no_mask"
    eval_data_path: str = (
        "dataset/kilt/qa_dataset/nq_augmented/natural_np_free.test.pickle"
    )
    eval_ids2text: str = "dataset/kilt/qa_dataset/nq_augmented/test_ids2text.json"
    add_instruction: bool = True
    put_hardneg: bool = False
    oracle: bool = False
    do_swap: bool = False
    train_sample: int = -1
    val_sample: int = -1
    allow_noret: bool = False
    rag_ctx_q: bool = False
    rag_q_ctx: bool = False


@dataclass
class joint_kilt_dataset_rettoken_augmented_synthetic_instruction_random:
    dataset: str = "joint_kilt_dataset_augmented"
    train_data_path: str = (
        "dataset/kilt/qa_dataset/nq_augmented/natural_np_free.train.pickle"
    )
    train_ids2text: str = "dataset/kilt/qa_dataset/nq_augmented/ids2text.json"
    input_query: bool = True
    train_data_type: str = "synthetic_instruction_random"
    eval_data_type: str = "synthetic_instruction_random"
    loss_mask_context: str = "no_mask"
    eval_data_path: str = (
        "dataset/kilt/qa_dataset/nq_augmented/natural_np_free.test.pickle"
    )
    eval_ids2text: str = "dataset/kilt/qa_dataset/nq_augmented/test_ids2text.json"
    add_instruction: bool = True
    put_hardneg: bool = False
    oracle: bool = False
    do_swap: bool = False
    train_sample: int = -1
    val_sample: int = -1
    allow_noret: bool = False
    rag_ctx_q: bool = False
    rag_q_ctx: bool = False


@dataclass
class joint_kilt_dataset_augmented:
    dataset: str = "joint_kilt_dataset_augmented"
    train_data_path: str = (
        "dataset/kilt/qa_dataset/nq_augmented/natural_np_free.train.pickle"
    )
    train_ids2text: str = "dataset/kilt/qa_dataset/nq_augmented/ids2text.json"
    input_query: bool = True
    train_data_type: str = "pretrain"
    eval_data_type: str = "pretrain"
    loss_mask_context: str = "context"
    eval_data_path: str = (
        "dataset/kilt/qa_dataset/nq_augmented/natural_np_free.test.pickle"
    )
    eval_ids2text: str = "dataset/kilt/qa_dataset/nq_augmented/test_ids2text.json"
    add_instruction: bool = True
    put_hardneg: bool = False
    oracle: bool = False
    do_swap: bool = False
    train_sample: int = -1
    val_sample: int = -1
    allow_noret: bool = False
    rag_ctx_q: bool = False
    rag_q_ctx: bool = False


@dataclass
class joint_kilt_dataset_augmented_debug:
    dataset: str = "joint_kilt_dataset_augmented"
    train_data_path: str = (
        "dataset/kilt/qa_dataset/nq_augmented/natural_np_free.train.pickle"
    )
    train_ids2text: str = "dataset/kilt/qa_dataset/nq_augmented/ids2text.json"
    input_query: bool = True
    train_data_type: str = "pretrain"
    eval_data_type: str = "pretrain"
    loss_mask_context: str = "context"
    eval_data_path: str = (
        "dataset/kilt/qa_dataset/nq_augmented/natural_np_free.train.pickle"
    )
    eval_ids2text: str = "dataset/kilt/qa_dataset/nq_augmented/ids2text.json"
    add_instruction: bool = True
    put_hardneg: bool = False
    oracle: bool = False
    do_swap: bool = False
    train_sample: int = -1
    val_sample: int = -1
    allow_noret: bool = False
    rag_ctx_q: bool = False
    rag_q_ctx: bool = False


@dataclass
class wikipedia_pretrain_10000:
    dataset: str = "wikipedia_pretrain_10000"
    train_data_path: str = "dataset/pretraining/subset.summarized.pickle"
    train_ids2text: str = "dataset/kilt/qa_dataset/nq_augmented/ids2text.json"
    input_query: bool = True
    train_data_type: str = "pretrain"
    eval_data_type: str = "pretrain"
    loss_mask_context: str = "no_mask"
    eval_data_path: str = (
        "dataset/kilt/qa_dataset/nq_augmented/natural_np_free.train.pickle"
    )
    eval_ids2text: str = "dataset/kilt/qa_dataset/nq_augmented/ids2text.json"
    add_instruction: bool = True
    put_hardneg: bool = False
    oracle: bool = False
    do_swap: bool = False
    train_sample: int = -1
    val_sample: int = -1
    allow_noret: bool = False
    rag_ctx_q: bool = False
    rag_q_ctx: bool = False


@dataclass
class finetuning_dataset:
    dataset: str = "finetuning_dataset"
    train_data_path: str = (
        "dataset/finetuning_dataset/long/natural_np_free.train.pickle"
    )
    train_ids2text: str = "dataset/finetuning_dataset/long/ids2text.json"
    input_query: bool = True
    loss_mask_context: str = "context"
    eval_data_path: str = "dataset/finetuning_dataset/long/natural_np_free.test.pickle"
    eval_ids2text: str = "dataset/finetuning_dataset/long/test_ids2text.json"
    add_instruction: bool = True
    put_hardneg: bool = False
    oracle: bool = False
    do_swap: bool = False
    train_data_type: str = "pretrain"
    eval_data_type: str = "pretrain"
    train_sample: int = -1
    val_sample: int = -1
    allow_noret: bool = False
    rag_ctx_q: bool = False
    rag_q_ctx: bool = False


@dataclass
class selfrag_dataset:
    dataset: str = "selfrag_dataset"
    train_data_path: str = (
        "dataset/finetuning_dataset/selfrag/natural_np_free.train.pickle"
    )
    train_ids2text: str = "dataset/finetuning_dataset/selfrag/ids2text.json"
    input_query: bool = True
    loss_mask_context: str = "context"
    eval_data_path: str = (
        "dataset/finetuning_dataset/selfrag/natural_np_free.test.pickle"
    )
    eval_ids2text: str = "dataset/finetuning_dataset/selfrag/test_ids2text.json"
    add_instruction: bool = True
    put_hardneg: bool = False
    oracle: bool = False
    do_swap: bool = False
    train_data_type: str = "pretrain"
    eval_data_type: str = "pretrain"
    train_sample: int = -1
    val_sample: int = -1
    allow_noret: bool = False
    rag_ctx_q: bool = False
    rag_q_ctx: bool = False


@dataclass
class selfrag_multi_dataset:
    dataset: str = "selfrag_multi_dataset"
    train_data_path: str = (
        "dataset/finetuning_dataset/selfrag_multi/natural_np_free.train.pickle"
    )
    train_ids2text: str = "dataset/finetuning_dataset/selfrag_multi/ids2text.json"
    input_query: bool = True
    loss_mask_context: str = "no_mask"
    eval_data_path: str = (
        "dataset/finetuning_dataset/selfrag_multi/natural_np_free.test.pickle"
    )
    eval_ids2text: str = "dataset/finetuning_dataset/selfrag_multi/test_ids2text.json"
    add_instruction: bool = True
    put_hardneg: bool = False
    oracle: bool = False
    do_swap: bool = False
    train_data_type: str = "pretrain"
    eval_data_type: str = "pretrain"
    train_sample: int = -1
    val_sample: int = -1
    allow_noret: bool = False
    rag_ctx_q: bool = False
    rag_q_ctx: bool = False
    mask_loss: bool = False


@dataclass
class selfrag_two_dataset:
    dataset: str = "selfrag_multi_dataset"
    train_data_path: str = (
        "dataset/finetuning_dataset/selfrag_multi/natural_np_free.train2.pickle"
    )
    train_ids2text: str = "dataset/finetuning_dataset/selfrag_multi/ids2text.json"
    input_query: bool = True
    loss_mask_context: str = "no_mask"
    eval_data_path: str = (
        "dataset/finetuning_dataset/selfrag_multi/natural_np_free.test.pickle"
    )
    eval_ids2text: str = "dataset/finetuning_dataset/selfrag_multi/test_ids2text.json"
    add_instruction: bool = True
    put_hardneg: bool = False
    oracle: bool = False
    do_swap: bool = False
    train_data_type: str = "pretrain"
    eval_data_type: str = "pretrain"
    train_sample: int = -1
    val_sample: int = -1
    allow_noret: bool = False
    rag_ctx_q: bool = False
    rag_q_ctx: bool = False
    mask_loss: bool = False

@dataclass
class selfrag_one_dataset:
    dataset: str = "selfrag_multi_dataset"
    train_data_path: str = (
        "dataset/finetuning_dataset/selfrag_multi/natural_np_free.train1.pickle"
    )
    train_ids2text: str = "dataset/finetuning_dataset/selfrag_multi/ids2text.json"
    input_query: bool = True
    loss_mask_context: str = "no_mask"
    eval_data_path: str = (
        "dataset/finetuning_dataset/selfrag_multi/natural_np_free.test.pickle"
    )
    eval_ids2text: str = "dataset/finetuning_dataset/selfrag_multi/test_ids2text.json"
    add_instruction: bool = True
    put_hardneg: bool = False
    oracle: bool = False
    do_swap: bool = False
    train_data_type: str = "pretrain"
    eval_data_type: str = "pretrain"
    train_sample: int = -1
    val_sample: int = -1
    allow_noret: bool = False
    rag_ctx_q: bool = False
    rag_q_ctx: bool = False
    mask_loss: bool = False



@dataclass
class selfrag_multi_dataset_split_loss:
    dataset: str = "selfrag_multi_dataset_split_loss"
    train_data_path: str = "dataset/finetuning_dataset/selfrag_multi/split.natural_np_free.train.pickle"
    train_ids2text: str = "dataset/finetuning_dataset/selfrag_multi/ids2text.json"
    input_query: bool= True
    loss_mask_context: str="no_mask"
    eval_data_path: str= "dataset/finetuning_dataset/selfrag_multi/split.natural_np_free.test.pickle"
    eval_ids2text: str = "dataset/finetuning_dataset/selfrag_multi/test_ids2text.json"
    add_instruction: bool=True
    put_hardneg: bool = False
    oracle: bool = False
    do_swap: bool = False
    train_data_type: str= "pretrain"
    eval_data_type: str= "pretrain"
    train_sample: int = -1
    val_sample: int = -1
    allow_noret: bool = False
    rag_ctx_q: bool = False
    rag_q_ctx: bool = False

@dataclass
class selfrag_multi_dataset_addq:
    dataset: str = "selfrag_multi_dataset_addq"
    train_data_path: str = "dataset/finetuning_dataset/selfrag_multi/add_q.natural_np_free.train.pickle"
    train_ids2text: str = "dataset/finetuning_dataset/selfrag_multi/ids2text.json"
    input_query: bool= True
    loss_mask_context: str="no_mask"
    eval_data_path: str= "dataset/finetuning_dataset/selfrag_multi/natural_np_free.test.pickle"
    eval_ids2text: str = "dataset/finetuning_dataset/selfrag_multi/test_ids2text.json"
    add_instruction: bool=True
    put_hardneg: bool = False
    oracle: bool = False
    do_swap: bool = False
    train_data_type: str= "pretrain"
    eval_data_type: str= "pretrain"
    train_sample: int = -1
    val_sample: int = -1
    allow_noret: bool = False
    rag_ctx_q: bool = False
    rag_q_ctx: bool = False

@dataclass
class hotpot_train_dataset:
    dataset: str = "hotpot_train_dataset"
    train_data_path: str = (
        "dataset/finetuning_dataset/hotpotqa/natural_np_free.train.pickle"
    )
    train_ids2text: str = "dataset/finetuning_dataset/hotpotqa/ids2text.json"
    input_query: bool = True
    loss_mask_context: str = "no_mask"
    eval_data_path: str = (
        "dataset/finetuning_dataset/hotpotqa/natural_np_free.test.pickle"
    )
    eval_ids2text: str = "dataset/finetuning_dataset/hotpotqa/test_ids2text.json"
    add_instruction: bool = True
    put_hardneg: bool = False
    oracle: bool = False
    do_swap: bool = False
    train_data_type: str = "pretrain"
    eval_data_type: str = "pretrain"
    train_sample: int = -1
    val_sample: int = -1
    allow_noret: bool = False
    rag_ctx_q: bool = False
    rag_q_ctx: bool = False


@dataclass
class selfrag_multi_dataset_implicit:
    dataset: str = "selfrag_multi_dataset_implicit"
    train_data_path: str = (
        "dataset/finetuning_dataset/selfrag_multi_implicit/natural_np_free.train.pickle"
    )
    train_ids2text: str = (
        "dataset/finetuning_dataset/selfrag_multi_implicit/ids2text.json"
    )
    input_query: bool = True
    loss_mask_context: str = "no_mask"
    eval_data_path: str = (
        "dataset/finetuning_dataset/selfrag_multi_implicit/natural_np_free.test.pickle"
    )
    eval_ids2text: str = (
        "dataset/finetuning_dataset/selfrag_multi_implicit/test_ids2text.json"
    )
    add_instruction: bool = True
    put_hardneg: bool = False
    oracle: bool = False
    do_swap: bool = False
    train_data_type: str = "pretrain"
    eval_data_type: str = "pretrain"
    train_sample: int = -1
    val_sample: int = -1
    allow_noret: bool = False
    rag_ctx_q: bool = False
    rag_q_ctx: bool = False


@dataclass
class selfrag_multi_dataset_addsingle:
    dataset: str = "selfrag_multi_dataset_addsingle"
    train_data_path: str = "dataset/finetuning_dataset/selfrag_multi_addsingle/natural_np_free.train.pickle"
    train_ids2text: str = (
        "dataset/finetuning_dataset/selfrag_multi_addsingle/ids2text.json"
    )
    input_query: bool = True
    loss_mask_context: str = "no_mask"
    eval_data_path: str = (
        "dataset/finetuning_dataset/selfrag_multi_addsingle/natural_np_free.test.pickle"
    )
    eval_ids2text: str = (
        "dataset/finetuning_dataset/selfrag_multi_addsingle/test_ids2text.json"
    )
    add_instruction: bool = True
    put_hardneg: bool = False
    oracle: bool = False
    do_swap: bool = False
    train_data_type: str = "pretrain"
    eval_data_type: str = "pretrain"
    train_sample: int = -1
    val_sample: int = -1
    allow_noret: bool = False
    rag_ctx_q: bool = False
    rag_q_ctx: bool = False


@dataclass
class alce_asqa:
    dataset: str = "alce_asqa"
    eval_data: str = "dataset/eval_dataset/ALCE-data/asqa_eval_gtr_top100.json"
    eval_docs: str = ""
    ndocs: int = 100
    ctx_truncate: bool = False
    task: str = "asqa"
    max_new_tokens: int = 300
    rag_ctx_q: bool = False
    rag_q_ctx: bool = False


@dataclass
class alce_eli5:
    dataset: str = "alce_eli5"
    eval_data: str = "dataset/eval_dataset/ALCE-data/eli5_eval_bm25_top100.w_id.json"
    eval_docs: str = ""
    ndocs: int = 100
    ctx_truncate: bool = False
    task: str = "eli5"
    max_new_tokens: int = 300
    rag_ctx_q: bool = False
    rag_q_ctx: bool = False


@dataclass
class alce_qampari:
    dataset: str = "alce_qampari"
    eval_data: str = "dataset/eval_dataset/ALCE-data/qampari_eval_gtr_top100.json"
    eval_docs: str = ""
    ndocs: int = 100
    ctx_truncate: bool = False
    task: str = "qampari"
    max_new_tokens: int = 300
    rag_ctx_q: bool = False
    rag_q_ctx: bool = False


@dataclass
class alce_musique:
    dataset: str = "alce_musique"
    eval_data: str = "dataset/eval/musique/musique_ans_dev_processed.json"
    eval_docs: str = ""
    ndocs: int = 100
    ctx_truncate: bool = False
    task: str = "musique"
    max_new_tokens: int = 300
    rag_ctx_q: bool = False
    rag_q_ctx: bool = False


@dataclass
class alce_musique_no_inst:
    dataset: str = "alce_musique"
    eval_data: str = "dataset/eval/musique/musique_ans_dev_processed.json"
    eval_docs: str = ""
    ndocs: int = 100
    ctx_truncate: bool = False
    task: str = "musique_no_inst"
    max_new_tokens: int = 300
    rag_ctx_q: bool = False
    rag_q_ctx: bool = False


@dataclass
class alce_strategyqa:
    dataset: str = "alce_strategyqa"
    eval_data: str = "dataset/eval/strategyqa/strategyqa_train_processed.json"
    eval_docs: str = "dataset/eval/strategyqa/strategyqa_train_processed_docs.json"
    ndocs: int = 100
    ctx_truncate: bool = False
    task: str = "strategyqa"
    max_new_tokens: int = 300
    rag_ctx_q: bool = False
    rag_q_ctx: bool = False


@dataclass
class alce_strategyqa_no_inst:
    dataset: str = "alce_strategyqa"
    eval_data: str = "dataset/eval/strategyqa/strategyqa_train_processed.json"
    eval_docs: str = "dataset/eval/strategyqa/strategyqa_train_processed_docs.json"
    ndocs: int = 100
    ctx_truncate: bool = False
    task: str = "strategyqa_no_inst"
    max_new_tokens: int = 300
    rag_ctx_q: bool = False
    rag_q_ctx: bool = False


@dataclass
class alce_hotpotqa_distractor:
    dataset: str = "alce_hotpotqa_distractor"
    eval_data: str = (
        "dataset/eval/hotpotqa/hotpotqa_dev_processed_distractor_paragraph.json"
    )
    eval_docs: str = ""
    ndocs: int = 100
    ctx_truncate: bool = False
    task: str = "hotpotqa_distractor"
    max_new_tokens: int = 300
    rag_ctx_q: bool = False
    rag_q_ctx: bool = False


@dataclass
class alce_hotpotqa_distractor_no_inst:
    dataset: str = "alce_hotpotqa_distractor"
    eval_data: str = (
        "dataset/eval/hotpotqa/hotpotqa_dev_processed_distractor_paragraph.json"
    )
    eval_docs: str = ""
    ndocs: int = 100
    ctx_truncate: bool = False
    task: str = "hotpotqa_distractor_no_inst"
    max_new_tokens: int = 300
    rag_ctx_q: bool = False
    rag_q_ctx: bool = False


@dataclass
class alce_hotpotqa_full_no_inst:
    dataset: str = "alce_hotpotqa_full"
    eval_data: str = "dataset/eval/hotpotqa/hotpotqa_dev_processed_fullwiki.json"
    eval_docs: str = "dataset/eval/hotpotqa/hotpotqa_dev_processed_fullwiki_docs.json"
    ndocs: int = 100
    ctx_truncate: bool = False
    task: str = "hotpotqa_full_no_inst"
    max_new_tokens: int = 300
    rag_ctx_q: bool = False
    rag_q_ctx: bool = False


@dataclass
class alce_hotpotqa_full:
    dataset: str = "alce_hotpotqa_full"
    eval_data: str = "dataset/eval/hotpotqa/hotpotqa_dev_processed_fullwiki.json"
    eval_docs: str = "dataset/eval/hotpotqa/hotpotqa_dev_processed_fullwiki_docs.json"
    ndocs: int = 100
    ctx_truncate: bool = False
    task: str = "hotpotqa_full"
    max_new_tokens: int = 300
    rag_ctx_q: bool = False
    rag_q_ctx: bool = False


@dataclass
class add_nq:
    dataset: str = "kilt_nq"
    eval_data: str= "/net/nfs.cirrascale/s2-research/amyl/sentence-decoding/dataset/eval/add/add.n_done.nq.contriever_msmarco.top100.json"
    eval_docs: str= ""
    ndocs: int = 100
    ctx_truncate: bool = False
    task: str = "nq"
    max_new_tokens: int = 300
    rag_ctx_q: bool = False
    rag_q_ctx: bool = False

@dataclass
class add_wow:
    dataset: str = "add_wow"
    eval_data: str= "/data/hyunji/KILT/add/add.n_done.wow.contriever_msmarco.top100.json"
    eval_docs: str= ""
    ndocs: int = 100
    ctx_truncate: bool = False
    task: str = "wow"
    max_new_tokens: int = 300
    rag_ctx_q: bool = False
    rag_q_ctx: bool = False

@dataclass
class add_fever:
    dataset: str = "add_fever"
    eval_data: str= "/data/hyunji/KILT/add/add.n_done.fever.contriever_msmarco.top100.json"
    eval_docs: str= ""
    ndocs: int = 100
    ctx_truncate: bool = False
    task: str = "fever"
    max_new_tokens: int = 300
    rag_ctx_q: bool = False
    rag_q_ctx: bool = False

@dataclass
class add_trex:
    dataset: str = "add_trex"
    eval_data: str= "/data/hyunji/KILT/add/add.n_done.trex.contriever_msmarco.top100.json"
    eval_docs: str= ""
    ndocs: int = 100
    ctx_truncate: bool = False
    task: str = "trex"
    max_new_tokens: int = 300
    rag_ctx_q: bool = False
    rag_q_ctx: bool = False

@dataclass
class add_zsre:
    dataset: str = "add_zsre"
    eval_data: str= "/data/hyunji/KILT/add/add.n_done.zsre.contriever_msmarco.top100.json"
    eval_docs: str= ""
    ndocs: int = 100
    ctx_truncate: bool = False
    task: str = "zsre"
    max_new_tokens: int = 300
    rag_ctx_q: bool = False
    rag_q_ctx: bool = False


@dataclass
class add_triviaqa:
    dataset: str = "add_triviaqa"
    eval_data: str= "/data/hyunji/KILT/add/add.n_done.triviaqa.contriever_msmarco.top100.json"
    eval_docs: str= ""
    ndocs: int = 100
    ctx_truncate: bool = False
    task: str = "triviaqa"
    max_new_tokens: int = 300
    rag_ctx_q: bool = False
    rag_q_ctx: bool = False


@dataclass
class add_eli5:
    dataset: str = "add_eli5"
    eval_data: str= "/data/hyunji/KILT/add/add.n_done.eli5.contriever_msmarco.top100.json"
    eval_docs: str= ""
    ndocs: int = 100
    ctx_truncate: bool = False
    task: str = "eli5"
    max_new_tokens: int = 300
    rag_ctx_q: bool = False
    rag_q_ctx: bool = False


@dataclass
class kilt_hotpotqa:
    dataset: str = "kilt_hotpotqa"
    eval_data: str= "/net/nfs.cirrascale/s2-research/amyl/sentence-decoding/dataset/eval/kilt-hotpotqa/n_done.hotpotqa.contriever_msmarco.top100.json"
    eval_docs: str= ""
    ndocs: int = 100
    ctx_truncate: bool = False
    task: str = "hotpotqa"
    max_new_tokens: int = 300
    rag_ctx_q: bool = False
    rag_q_ctx: bool = False


class add_fever:
    dataset: str = "kilt_fever"
    eval_data: str= "/net/nfs.cirrascale/s2-research/amyl/sentence-decoding/dataset/eval/add/add.n_done.fever.contriever_msmarco.top100.json"
    eval_docs: str= ""
    ndocs: int = 100
    ctx_truncate: bool = False
    task: str = "fever"
    max_new_tokens: int = 300
    rag_ctx_q: bool = False
    rag_q_ctx: bool = False

@dataclass
class add_trex:
    dataset: str = "kilt_trex"
    eval_data: str= "/net/nfs.cirrascale/s2-research/amyl/sentence-decoding/dataset/eval/add/add.n_done.trex.contriever_msmarco.top100.json"
    eval_docs: str= ""
    ndocs: int = 100
    ctx_truncate: bool = False
    task: str = "trex"
    max_new_tokens: int = 300
    rag_ctx_q: bool = False
    rag_q_ctx: bool = False

@dataclass
class add_zsre:
    dataset: str = "kilt_zsre"
    eval_data: str= "/net/nfs.cirrascale/s2-research/amyl/sentence-decoding/dataset/eval/add/add.n_done.zsre.contriever_msmarco.top100.json"
    eval_docs: str= ""
    ndocs: int = 100
    ctx_truncate: bool = False
    task: str = "zsre"
    max_new_tokens: int = 300
    rag_ctx_q: bool = False
    rag_q_ctx: bool = False

@dataclass
class add_wow:
    dataset: str = "kilt_wow"
    eval_data: str= "/net/nfs.cirrascale/s2-research/amyl/sentence-decoding/dataset/eval/add/add.n_done.wow.contriever_msmarco.top100.json"
    eval_docs: str= ""
    ndocs: int = 100
    ctx_truncate: bool = False
    task: str = "wow"
    max_new_tokens: int = 300
    rag_ctx_q: bool = False
    rag_q_ctx: bool = False

@dataclass
class oracle_nq:
    dataset: str = "kilt_nq"
    eval_data: str= "dataset/eval/kilt-nq/oracle.nq.contriever_msmarco.top100.json"
    eval_docs: str= ""
    ndocs: int = 1
    ctx_truncate: bool = False
    task: str = "nq"
    max_new_tokens: int = 300
    rag_ctx_q: bool = False
    rag_q_ctx: bool = False


@dataclass
class oracle_hotpotqa:
    dataset: str = "kilt_hotpotqa"
    eval_data: str= "dataset/eval/kilt-hotpotqa/oracle.hotpotqa.contriever_msmarco.top100.json"
    eval_docs: str= ""
    ndocs: int = 1
    ctx_truncate: bool = False
    task: str = "hotpotqa"
    max_new_tokens: int = 300
    rag_ctx_q: bool = False
    rag_q_ctx: bool = False



@dataclass
class oracle_fever:
    dataset: str = "kilt_fever"
    eval_data: str = "dataset/eval/kilt-fever/oracle.fever.contriever_msmarco.top100.json"
    eval_docs: str = ""
    ndocs: int = 100
    ctx_truncate: bool = False
    task: str = "fever"
    max_new_tokens: int = 300
    rag_ctx_q: bool = False
    rag_q_ctx: bool = False


@dataclass
class oracle_wow:
    dataset: str = "kilt_wow"
    eval_data: str = "dataset/eval/kilt-wow/oracle.wow.contriever_msmarco.top100.json"
    eval_docs: str = ""
    ndocs: int = 100
    ctx_truncate: bool = False
    task: str = "wow"
    max_new_tokens: int = 300
    rag_ctx_q: bool = False
    rag_q_ctx: bool = False

@dataclass
class oracle_zsre:
    dataset: str = "kilt_zsre"
    eval_data: str = "dataset/eval/kilt-zsre/oracle.zsre.contriever_msmarco.top100.json"
    eval_docs: str = ""
    ndocs: int = 100
    ctx_truncate: bool = False
    task: str = "zsre"
    max_new_tokens: int = 300
    rag_ctx_q: bool = False
    rag_q_ctx: bool = False


@dataclass
class oracle_trex:
    dataset: str = "kilt_trex"
    eval_data: str = "dataset/eval/kilt-trex/oracle.trex.contriever_msmarco.top100.json"
    eval_docs: str = ""
    ndocs: int = 100
    ctx_truncate: bool = False
    task: str = "trex"
    max_new_tokens: int = 300
    rag_ctx_q: bool = False
    rag_q_ctx: bool = False

@dataclass
class oracle_zsre:
    dataset: str = "kilt_zsre"
    eval_data: str = "dataset/eval/kilt-zsre/oracle.zsre.contriever_msmarco.top100.json"
    eval_docs: str = ""
    ndocs: int = 100
    ctx_truncate: bool = False
    task: str = "zsre"
    max_new_tokens: int = 300
    rag_ctx_q: bool = False
    rag_q_ctx: bool = False

@dataclass
class oracle_nq:
    dataset: str = "kilt_nq"
    eval_data: str= "dataset/eval/kilt-nq/oracle.nq.contriever_msmarco.top100.json"
    eval_docs: str= ""
    ndocs: int = 1
    ctx_truncate: bool = False
    task: str = "nq"
    max_new_tokens: int = 300
    rag_ctx_q: bool = False
    rag_q_ctx: bool = False


@dataclass
class oracle_hotpotqa:
    dataset: str = "kilt_hotpotqa"
    eval_data: str= "dataset/eval/kilt-hotpotqa/oracle.hotpotqa.contriever_msmarco.top100.json"
    eval_docs: str= ""
    ndocs: int = 1
    ctx_truncate: bool = False
    task: str = "hotpotqa"
    max_new_tokens: int = 300
    rag_ctx_q: bool = False
    rag_q_ctx: bool = False



@dataclass
class oracle_fever:
    dataset: str = "kilt_fever"
    eval_data: str = "dataset/eval/kilt-fever/oracle.fever.contriever_msmarco.top100.json"
    eval_docs: str = ""
    ndocs: int = 100
    ctx_truncate: bool = False
    task: str = "fever"
    max_new_tokens: int = 300
    rag_ctx_q: bool = False
    rag_q_ctx: bool = False


@dataclass
class oracle_wow:
    dataset: str = "kilt_wow"
    eval_data: str = "dataset/eval/kilt-wow/oracle.wow.contriever_msmarco.top100.json"
    eval_docs: str = ""
    ndocs: int = 100
    ctx_truncate: bool = False
    task: str = "wow"
    max_new_tokens: int = 300
    rag_ctx_q: bool = False
    rag_q_ctx: bool = False

@dataclass
class oracle_zsre:
    dataset: str = "kilt_zsre"
    eval_data: str = "dataset/eval/kilt-zsre/oracle.zsre.contriever_msmarco.top100.json"
    eval_docs: str = ""
    ndocs: int = 100
    ctx_truncate: bool = False
    task: str = "zsre"
    max_new_tokens: int = 300
    rag_ctx_q: bool = False
    rag_q_ctx: bool = False


@dataclass
class oracle_trex:
    dataset: str = "kilt_trex"
    eval_data: str = "dataset/eval/kilt-trex/oracle.trex.contriever_msmarco.top100.json"
    eval_docs: str = ""
    ndocs: int = 100
    ctx_truncate: bool = False
    task: str = "trex"
    max_new_tokens: int = 300
    rag_ctx_q: bool = False
    rag_q_ctx: bool = False

@dataclass
class oracle_zsre:
    dataset: str = "kilt_zsre"
    eval_data: str = "dataset/eval/kilt-zsre/oracle.zsre.contriever_msmarco.top100.json"
    eval_docs: str = ""
    ndocs: int = 100
    ctx_truncate: bool = False
    task: str = "zsre"
    max_new_tokens: int = 300
    rag_ctx_q: bool = False
    rag_q_ctx: bool = False


@dataclass
class oracle_trex:
    dataset: str = "kilt_trex"
    eval_data: str = "dataset/eval/kilt-trex/oracle.trex.contriever_msmarco.top100.json"
    eval_docs: str = ""
    ndocs: int = 100
    ctx_truncate: bool = False
    task: str = "trex"
    max_new_tokens: int = 300
    rag_ctx_q: bool = False
    rag_q_ctx: bool = False

@dataclass
class oracle_eli5:
    dataset: str = "kilt_eli5"
    eval_data: str = "dataset/eval/kilt-eli5/oracle.eli5.contriever_msmarco.top100.json"
    eval_docs: str = ""
    ndocs: int = 100
    ctx_truncate: bool = False
    task: str = "eli5"
    max_new_tokens: int = 300
    rag_ctx_q: bool = False
    rag_q_ctx: bool = False


@dataclass
class oracle_triviaqa:
    dataset: str = "kilt_triviaqa"
    eval_data: str = "dataset/eval/kilt-triviaqa/oracle.triviaqa.contriever_msmarco.top100.json"
    eval_docs: str = ""
    ndocs: int = 100
    ctx_truncate: bool = False
    task: str = "triviaqa"
    max_new_tokens: int = 300
    rag_ctx_q: bool = False
    rag_q_ctx: bool = False

@dataclass
class wrong_nq:
    dataset: str = "kilt_nq"
    eval_data: str= "dataset/eval/kilt-nq/wrong.nq.contriever_msmarco.top100.json"
    eval_docs: str= ""
    ndocs: int = 1
    ctx_truncate: bool = False
    task: str = "nq"
    max_new_tokens: int = 300
    rag_ctx_q: bool = False
    rag_q_ctx: bool = False


@dataclass
class wrong_fever:
    dataset: str = "kilt_fever"
    eval_data: str = "dataset/eval/kilt-fever/wrong.fever.contriever_msmarco.top100.json"
    eval_docs: str = ""
    ndocs: int = 1
    ctx_truncate: bool = False
    task: str = "fever"
    max_new_tokens: int = 300
    rag_ctx_q: bool = False
    rag_q_ctx: bool = False


@dataclass
class wrong_wow:
    dataset: str = "kilt_wow"
    eval_data: str = "dataset/eval/kilt-wow/wrong.wow.contriever_msmarco.top100.json"
    eval_docs: str = ""
    ndocs: int = 1
    ctx_truncate: bool = False
    task: str = "wow"
    max_new_tokens: int = 300
    rag_ctx_q: bool = False
    rag_q_ctx: bool = False

@dataclass
class wrong_zsre:
    dataset: str = "kilt_zsre"
    eval_data: str = "dataset/eval/kilt-zsre/wrong.zsre.contriever_msmarco.top100.json"
    eval_docs: str = ""
    ndocs: int = 1
    ctx_truncate: bool = False
    task: str = "zsre"
    max_new_tokens: int = 300
    rag_ctx_q: bool = False
    rag_q_ctx: bool = False


@dataclass
class wrong_trex:
    dataset: str = "kilt_trex"
    eval_data: str = "dataset/eval/kilt-trex/wrong.trex.contriever_msmarco.top100.json"
    eval_docs: str = ""
    ndocs: int = 1
    ctx_truncate: bool = False
    task: str = "trex"
    max_new_tokens: int = 300
    rag_ctx_q: bool = False
    rag_q_ctx: bool = False

@dataclass
class wrong_zsre:
    dataset: str = "kilt_zsre"
    eval_data: str = "dataset/eval/kilt-zsre/wrong.zsre.contriever_msmarco.top100.json"
    eval_docs: str = ""
    ndocs: int = 1
    ctx_truncate: bool = False
    task: str = "zsre"
    max_new_tokens: int = 300
    rag_ctx_q: bool = False
    rag_q_ctx: bool = False


@dataclass
class wrong_trex:
    dataset: str = "kilt_trex"
    eval_data: str = "dataset/eval/kilt-trex/wrong.trex.contriever_msmarco.top100.json"
    eval_docs: str = ""
    ndocs: int = 1
    ctx_truncate: bool = False
    task: str = "trex"
    max_new_tokens: int = 300
    rag_ctx_q: bool = False
    rag_q_ctx: bool = False

@dataclass
class wrong_eli5:
    dataset: str = "kilt_eli5"
    eval_data: str = "dataset/eval/kilt-eli5/wrong.eli5.contriever_msmarco.top100.json"
    eval_docs: str = ""
    ndocs: int = 1
    ctx_truncate: bool = False
    task: str = "eli5"
    max_new_tokens: int = 300
    rag_ctx_q: bool = False
    rag_q_ctx: bool = False


@dataclass
class wrong_triviaqa:
    dataset: str = "kilt_triviaqa"
    eval_data: str = "dataset/eval/kilt-triviaqa/wrong.triviaqa.contriever_msmarco.top100.json"
    eval_docs: str = ""
    ndocs: int = 1
    ctx_truncate: bool = False
    task: str = "triviaqa"
    max_new_tokens: int = 300
    rag_ctx_q: bool = False
    rag_q_ctx: bool = False


@dataclass
class oracle_trex:
    dataset: str = "kilt_trex"
    eval_data: str = "dataset/eval/kilt-trex/oracle.trex.contriever_msmarco.top100.json"
    eval_docs: str = ""
    ndocs: int = 100
    ctx_truncate: bool = False
    task: str = "trex"
    max_new_tokens: int = 300
    rag_ctx_q: bool = False
    rag_q_ctx: bool = False

@dataclass
class oracle_eli5:
    dataset: str = "kilt_eli5"
    eval_data: str = "dataset/eval/kilt-eli5/oracle.eli5.contriever_msmarco.top100.json"
    eval_docs: str = ""
    ndocs: int = 100
    ctx_truncate: bool = False
    task: str = "eli5"
    max_new_tokens: int = 300
    rag_ctx_q: bool = False
    rag_q_ctx: bool = False


@dataclass
class oracle_triviaqa:
    dataset: str = "kilt_triviaqa"
    eval_data: str = "dataset/eval/kilt-triviaqa/oracle.triviaqa.contriever_msmarco.top100.json"
    eval_docs: str = ""
    ndocs: int = 100
    ctx_truncate: bool = False
    task: str = "triviaqa"
    max_new_tokens: int = 300
    rag_ctx_q: bool = False
    rag_q_ctx: bool = False

@dataclass
class kilt_nq:
    dataset: str = "kilt_nq"
    eval_data: str= "dataset/eval/kilt-nq/done.nq.contriever_msmarco.top100.json"
    #eval_data: str= "dataset/eval/kilt-nq/toy.json"
    eval_docs: str= ""
    ndocs: int = 100
    ctx_truncate: bool = False
    task: str = "nq"
    max_new_tokens: int = 300
    rag_ctx_q: bool = False
    rag_q_ctx: bool = False

@dataclass
class toy_kilt_nq:
    dataset: str = "kilt_nq"
    #eval_data: str= "dataset/eval/kilt-nq/done.nq.contriever_msmarco.top100.json"
    eval_data: str= "dataset/eval/kilt-nq/toy.json"
    eval_docs: str= ""
    ndocs: int = 100
    ctx_truncate: bool = False
    task: str = "nq"
    max_new_tokens: int = 300
    rag_ctx_q: bool = False
    rag_q_ctx: bool = False


@dataclass
class kilt_fever:
    dataset: str = "kilt_fever"
    eval_data: str = "dataset/eval/kilt-fever/done.fever.contriever_msmarco.top100.json"
    eval_docs: str = ""
    ndocs: int = 100
    ctx_truncate: bool = False
    task: str = "fever"
    max_new_tokens: int = 300
    rag_ctx_q: bool = False
    rag_q_ctx: bool = False


@dataclass
class kilt_wow:
    dataset: str = "kilt_wow"
    eval_data: str = "dataset/eval/kilt-wow/done.wow.contriever_msmarco.top100.json"
    eval_docs: str = ""
    ndocs: int = 100
    ctx_truncate: bool = False
    task: str = "wow"
    max_new_tokens: int = 300
    rag_ctx_q: bool = False
    rag_q_ctx: bool = False

@dataclass
class kilt_zsre:
    dataset: str = "kilt_zsre"
    eval_data: str = "dataset/eval/kilt-zsre/n_done.zsre.contriever_msmarco.top100.json"
    eval_docs: str = ""
    ndocs: int = 100
    ctx_truncate: bool = False
    task: str = "zsre"
    max_new_tokens: int = 300
    rag_ctx_q: bool = False
    rag_q_ctx: bool = False


@dataclass
class kilt_trex:
    dataset: str = "kilt_trex"
    eval_data: str = "dataset/eval/kilt-trex/n_done.trex.contriever_msmarco.top100.json"
    eval_docs: str = ""
    ndocs: int = 100
    ctx_truncate: bool = False
    task: str = "trex"
    max_new_tokens: int = 300
    rag_ctx_q: bool = False
    rag_q_ctx: bool = False

@dataclass
class kilt_zsre:
    dataset: str = "kilt_zsre"
    eval_data: str = "dataset/eval/kilt-zsre/n_done.zsre.contriever_msmarco.top100.json"
    eval_docs: str = ""
    ndocs: int = 100
    ctx_truncate: bool = False
    task: str = "zsre"
    max_new_tokens: int = 300
    rag_ctx_q: bool = False
    rag_q_ctx: bool = False


@dataclass
class kilt_trex:
    dataset: str = "kilt_trex"
    eval_data: str = "dataset/eval/kilt-trex/n_done.trex.contriever_msmarco.top100.json"
    eval_docs: str = ""
    ndocs: int = 100
    ctx_truncate: bool = False
    task: str = "trex"
    max_new_tokens: int = 300
    rag_ctx_q: bool = False
    rag_q_ctx: bool = False

@dataclass
class kilt_eli5:
    dataset: str = "kilt_eli5"
    eval_data: str = "dataset/eval/kilt-eli5/n_done.eli5.contriever_msmarco.top100.json"
    eval_docs: str = ""
    ndocs: int = 100
    ctx_truncate: bool = False
    task: str = "eli5"
    max_new_tokens: int = 300
    rag_ctx_q: bool = False
    rag_q_ctx: bool = False


@dataclass
class kilt_triviaqa:
    dataset: str = "kilt_triviaqa"
    eval_data: str = "dataset/eval/kilt-triviaqa/n_done.triviaqa.contriever_msmarco.top100.json"
    eval_docs: str = ""
    ndocs: int = 100
    ctx_truncate: bool = False
    task: str = "triviaqa"
    max_new_tokens: int = 300
    rag_ctx_q: bool = False
    rag_q_ctx: bool = False

@dataclass
class vanilla_nq:
    dataset: str = "kilt_nq"
    eval_data: str = "fine-tuned_double_selfrag_multi_vanilla/kilt_nq_top20_ret.json"
    eval_docs: str = ""
    ndocs: int = 100
    ctx_truncate: bool = False
    task: str = "nq"
    max_new_tokens: int = 300
    rag_ctx_q: bool = False
    rag_q_ctx: bool = False
