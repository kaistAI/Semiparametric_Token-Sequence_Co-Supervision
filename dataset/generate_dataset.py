import torch
from torch.utils.data import Dataset
from typing import Optional, Dict, Sequence
from dataclasses import dataclass, field

import transformers
import json, copy
from tqdm import tqdm
from utils.dataset_utils import add_instruction_rettoken
IGNORE_INDEX = -100
TASK_INST = {
    "wow": "Given a chat history separated by new lines, generates an informative, knowledgeable and engaging response. ",
    #  "fever": "Is the following statement correct or not? Say true if it's correct; otherwise say false",
    "fever": "Is the following statement correct or not? Say supports if it's correct; otherwise say refutes. Also provide the evidence.",
    "eli5": "Provide a paragraph-length response using simple words to answer the following question.",
    "nq": "Answer the following question with corresponding evidence.",
    "triviaqa": "Answer the following question with corresponding evidence.",
    "zsre": "Find the correct entity given subject and relation with corresponding evidence.",
    "trex": "Find the correct object given subject and relation with corresponding evidence.",
    "obqa": "Given four answer candidates, A, B, C and D, choose the best answer choice.",
    "arc_easy": "Given four answer candidates, A, B, C and D, choose the best answer choice.",
    "arc_c": "Given four answer candidates, A, B, C and D, choose the best answer choice.",
    "asqa": "Answer the following question. The question may be ambiguous and have multiple correct answers, and in that case, you have to provide a long-form answer including all correct answers.",
    "qampari": "Provide a list of accurate answers for the given question using only the provided search results (some of which might be irrelevant) and cite them properly. Always cite one and only one document for each answer. Separate answers by commas. For questions that have more than 5 answers, write at least 5 answers.",
    "musique": "Answer question that require multiple reasoning steps where each reasoning steps have corresponding evidence.",
    "strategyqa": "Answer question in which the required reasoning steps are implicit in the question.",
    "hotpotqa_distractor": "Answer question that require 2 step reasoning where each reasoning steps have corresponding evidence.",
    "hotpotqa": "Answer question that require 2 step reasoning where each reasoning steps have corresponding evidence.",
    "hotpotqa_distractor_no_inst": "",
    "musique_no_inst": "",
    "hotpotqa_full": "Answer question that require 2 step reasoning where each reasoning steps have corresponding evidence.",
    "hotpotqa_full_no_inst": "",
    "strategyqa_no_inst": "",
}

def do_mask(type):
    if type == "no_mask":
        return False
    else:
        return True

def do_print(sen):
    # print(f"current device: {torch.cuda.current_device()}")
    if torch.cuda.current_device() == 0:
        print(sen)

def _tokenize_fn(
        strings: Sequence[str], 
        tokenizer: transformers.PreTrainedTokenizer, 
        padding_type="longest"
) -> Dict:
    tokenized_list = []

    cnt = 0

    for text in strings:
        # print(cnt, text)
        tokenized_list.append(
            tokenizer(
                text,
                return_tensors="pt",
                padding=padding_type, #"max_length", # longest
                max_length=tokenizer.model_max_length,
                truncation=True,
            )
        )
        # if cnt%1000==0:
        #     print(cnt, text)
        cnt += 1

    # print("tokenized_list", tokenized_list[0])
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    # print("input_ids", input_ids)
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item()
        for tokenized in tokenized_list
    ]
    # print("input_ids_lens", input_ids_lens)
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )

def preprocess(
    sources: Sequence[str],
    targets: Sequence[Sequence],
    hard_negs: Sequence[str],
    targets_whole: Sequence[str],
    # target_idxs: Sequence[str],
    hard_negs_whole: Sequence[str],
    ctx_tokenizer: transformers.PreTrainedTokenizer,
    question_tokenizer: transformers.PreTrainedTokenizer,
    loss_mask_context: str,
    data_path: str,
    padding_type="longest"
) -> Dict:
    """Preprocess the data by tokenizing."""
    if targets != None:
        examples = [s + t for s, t in zip(sources, targets)]
        do_print(f"### Example [{len(examples)}] in preprocess function: {examples[0:10]}")
    else:
        examples = sources

    examples_tokenized, sources_tokenized = [
        _tokenize_fn(strings, question_tokenizer, padding_type) for strings in (examples, sources)
    ]
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    target_ids = None
    hard_neg_ids = None

    datanum = len(labels)

    # if data_path.endswith(".pickle"):
    #     data_f = pickle.load(open(data_path, "rb"))
    # else:
    #     assert False

    if targets_whole is not None:
        target_ids = []
    if hard_negs_whole is not None:
        hard_neg_ids = []

    new_target_idxs = []
    for i in tqdm(range(datanum)):
        source_len = sources_tokenized["input_ids_lens"][i]
        new_target_idxs.append(
            torch.where(
                input_ids[i] == question_tokenizer.convert_tokens_to_ids("[Ret]")
            )[0].tolist()
        )
        if targets_whole is not None:
            target_idx = new_target_idxs[i]
            target_id = _tokenize_fn(targets_whole[i], ctx_tokenizer, padding_type)["input_ids"][
                : len(target_idx)
            ]
            # do_print(question_tokenizer.decode(input_ids[i]))
            assert len(target_id) == len(target_idx)
            target_ids.append(target_id)
        if hard_negs_whole is not None:
            hard_neg_id = _tokenize_fn(tqdm(hard_negs_whole[i]), ctx_tokenizer, padding_type)[
                "input_ids"
            ]
            hard_neg_ids.append(hard_neg_id)
        label = labels[i]

        input_id = input_ids[i]

        if i == 0: 
            # do_print(f"input_ids was .. {input_id}")
            do_print(f"label was ..) {label}")
        label[:source_len] = IGNORE_INDEX

        if do_mask(loss_mask_context):
            mask_idx = get_mask_idx(input_id, question_tokenizer)      
            label[mask_idx] = IGNORE_INDEX
            cnt = list(label).count(IGNORE_INDEX)
    print("Tokenizing All done!")
    return dict(
        input_ids=input_ids,
        labels=labels,
        target_ids=target_ids,
        target_idxs=new_target_idxs,
        hard_neg_ids=hard_neg_ids,
    )


def get_mask_idx(input_ids, tokenizer):

    mask_idx = []
    flag = 0
    for idx, item in enumerate(input_ids):
        if flag:
            mask_idx.append(idx)
        if item == tokenizer.convert_tokens_to_ids("[Cs]") and flag:
            flag = 0
        if item == tokenizer.convert_tokens_to_ids("[Ret]"):
            flag = 1
    return mask_idx


class ALCE_Top100_Dataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
        self,
        ctx_tokenizer: transformers.PreTrainedTokenizer,
        question_tokenizer: transformers.PreTrainedTokenizer,
        dataset_config,
    ):
        super(ALCE_Top100_Dataset, self).__init__()
        instructions = TASK_INST[dataset_config.task]
        input_path = dataset_config.eval_data
        if input_path.endswith(".json"):
            input_data = json.load(open(input_path))
        else:
            import jsonlines

            with jsonlines.open(input_path, "r") as jsonl_f:
                input_data = [obj for obj in jsonl_f]

        if dataset_config.rag_ctx_q:
            self.prompt = _tokenize_fn(
                tqdm(
                    [
                        instructions
                        + "## Context: [Ret]"
                        + item["docs"][0]["text"]
                        + "## Input:\n\n"
                        + item["question"]
                        + "## Output:\n\n"
                        for item in input_data
                    ]
                ),
                question_tokenizer,
            )["input_ids"]
        elif dataset_config.rag_q_ctx:
            self.prompt = _tokenize_fn(
                tqdm(
                    [
                        instructions
                        + "## Input:\n\n"
                        + item["question"]
                        + "## Context: [Ret]"
                        + item["docs"][0]["text"]
                        + "## Output:\n\n"
                        for item in input_data
                    ]
                ),
                question_tokenizer,
            )["input_ids"]
        else:
            self.prompt = _tokenize_fn(
                tqdm(
                    [
                        instructions
                        + "## Input:\n\n"
                        + item["question"]
                        + "## Output:\n\n"
                        for item in input_data
                    ]
                ),
                question_tokenizer,
            )["input_ids"]

        self.ids = [
            [doc["id"] for doc in item["docs"][: dataset_config.ndocs]]
            for item in input_data
        ]

    def __len__(self):
        return len(self.prompt)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(
            input_ids=self.prompt[i],
            target_labels=self.ids[i],
            alce_idx=i,
        )


class ALCE_Top100_FullDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
        self,
        ctx_tokenizer: transformers.PreTrainedTokenizer,
        question_tokenizer: transformers.PreTrainedTokenizer,
        dataset_config,
    ):
        super(ALCE_Top100_FullDataset, self).__init__()
        instructions = TASK_INST[dataset_config.task]
        input_path = dataset_config.eval_data
        if input_path.endswith(".json"):
            input_data = json.load(open(input_path))
        else:
            import jsonlines

            with jsonlines.open(input_path, "r") as jsonl_f:
                input_data = [obj for obj in jsonl_f]
        if dataset_config.rag_ctx_q:
            self.prompt = _tokenize_fn(
                tqdm(
                    [
                        instructions
                        + "## Context: [Ret]"
                        + item["docs"][0]["text"]
                        + "## Input:\n\n"
                        + item["question"]
                        + "## Output:\n\n"
                        for item in input_data
                    ]
                ),
                question_tokenizer,
            )["input_ids"]
        elif dataset_config.rag_q_ctx:
            self.prompt = _tokenize_fn(
                tqdm(
                    [
                        instructions
                        + "## Input:\n\n"
                        + item["question"]
                        + "## Context: [Ret]"
                        + item["docs"][0]["text"]
                        + "## Output:\n\n"
                        for item in input_data
                    ]
                ),
                question_tokenizer,
            )["input_ids"]
        else:
            self.prompt = _tokenize_fn(
                tqdm(
                    [
                        instructions
                        + "## Input:\n\n"
                        + item["question"]
                        + "## Output:\n\n"
                        for item in input_data
                    ]
                ),
                question_tokenizer,
            )["input_ids"]

        self.ids = []
        for elem in json.load(open(dataset_config.eval_docs)):
            self.ids.append(elem["id"])

    def __len__(self):
        return len(self.prompt)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(
            input_ids=self.prompt[i],
            target_labels=self.ids,
            alce_idx=i,
        )


class ALCE_Top100_FullEmbedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
        self,
        ctx_tokenizer: transformers.PreTrainedTokenizer,
        question_tokenizer: transformers.PreTrainedTokenizer,
        dataset_config,
    ):
        super(ALCE_Top100_FullEmbedDataset, self).__init__()
        input_path = dataset_config.eval_docs
        if input_path.endswith(".json"):
            input_data = json.load(open(input_path))
        else:
            assert False
            import jsonlines

            with jsonlines.open(input_path, "r") as jsonl_f:
                input_data = [obj for obj in jsonl_f]
        input_dict = {}
        for doc in input_data:
            if "title" in doc.keys():
                input_dict[doc["id"]] = f"{doc['title']} :: {doc['text']}"
            else:
                input_dict[doc["id"]] = doc["text"]
        print(f"# of doc: {len(input_dict)}")
        self.ids = list(input_dict.keys())
        self.ctx = _tokenize_fn(tqdm(list(input_dict.values())), ctx_tokenizer)[
            "input_ids"
        ]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(
            input_ids=self.ctx[i], target_labels=self.ids[i]
        )


class ALCE_Top100_EmbedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
        self,
        ctx_tokenizer: transformers.PreTrainedTokenizer,
        question_tokenizer: transformers.PreTrainedTokenizer,
        dataset_config,
    ):
        super(ALCE_Top100_EmbedDataset, self).__init__()
        input_path = dataset_config.eval_data
        if input_path.endswith(".json"):
            input_data = json.load(open(input_path))
        else:
            import jsonlines

            with jsonlines.open(input_path, "r") as jsonl_f:
                input_data = [obj for obj in jsonl_f]
        input_dict = {}
        for item in input_data:
            for doc in item["docs"][: dataset_config.ndocs]:
                input_dict[doc["id"]] = doc["text"]
        self.ids = list(input_dict.keys())
        self.ctx = _tokenize_fn(tqdm(list(input_dict.values())), ctx_tokenizer)[
            "input_ids"
        ]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(
            input_ids=self.ctx[i], target_labels=self.ids[i]
        )


class TrainDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
        self,
        ctx_tokenizer: transformers.PreTrainedTokenizer,
        question_tokenizer: transformers.PreTrainedTokenizer,
        dataset_config,
        # dataset_type,
    ):
        super(TrainDataset, self).__init__()
        data_path = dataset_config.train_data_path
        ids2text = json.load(open(dataset_config.train_ids2text))
        loss_mask_context = dataset_config.loss_mask_context
        list_data_dict = add_instruction_rettoken(
            data_path,
            dataset_config,
        ) 
        sources = [
            example["instruction"]
            for example in list_data_dict
        ]
        targets = [
            f"{example['output']}{question_tokenizer.eos_token}"
            for example in list_data_dict
        ]
        if targets != None:
            print(f"Target Ex: {targets[0]}")
            print(
                f"Target Avg Len: {sum([len(target) for target in targets])/len(targets)}"
            )
        target_labels = [example["evidence"] for example in tqdm(list_data_dict)]
        targets_whole = [
            [ids2text[item] for item in example["evidence"]]
            for example in tqdm(list_data_dict)
        ]
        # target_idxs -> [Ret] 위치
        # target_idxs = [example["target_idxs"] for example in tqdm(list_data_dict)]
        if dataset_config.put_hardneg:
            hard_neg_labels = [
                example["hard_negs"][0] if len(example["hard_negs"]) > 0 else ""
                for example in tqdm(list_data_dict)
            ]
            hard_negs = [
                example["hard_negs"][1] if len(example["hard_negs"]) > 0 else ""
                for example in tqdm(list_data_dict)
            ]
            hard_negs_whole = [
                [ids2text[item] for item in example["evidence"]]
                for example in tqdm(list_data_dict)
            ]

            print(f"HardNeg Ex: {hard_negs[0]}")
            print(
                f"HardNeg Avg Len: {sum([len(hard_neg) for hard_neg in hard_negs])/len(hard_negs)}"
            )
        else:
            hard_neg_labels, hard_negs_whole, hard_negs = None, None, None

        print("Tokenizing inputs... This may take some time...")

        if targets != None:
            do_print(f"## targets: {targets[0]}")
        else:
            do_print("## targets is None!")
        if sources != None:
            do_print(f"## sources: {sources[0]}")
        else:
            do_print("## is None!")

        data_dict = preprocess(
            sources,
            targets,
            hard_negs,
            targets_whole,
            # target_idxs,
            hard_negs_whole,
            ctx_tokenizer,
            question_tokenizer,
            loss_mask_context,
            data_path,
        )

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]
        self.target_ids = data_dict["target_ids"]
        self.target_idxs = data_dict["target_idxs"]

        self.hard_neg_ids = data_dict["hard_neg_ids"]
        self.target_labels = target_labels
        self.hard_neg_labels = hard_neg_labels
        do_print(
            f"####### Example dataset..\nsource: {sources[0]}\ninput_ids: {self.input_ids[0]}\nlabels: {self.labels[0]}\nmask_idx: {self.mask_idx}"
        )
        

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        if self.target_ids is None:
            return dict(
                input_ids=self.input_ids[i],
                labels=self.labels[i],
                target_labels=self.target_labels[i],
                mask_idx=self.mask_idx[i]
            )
        elif self.hard_neg_ids is None:
            return dict(
                input_ids=self.input_ids[i],
                labels=self.labels[i],
                target_ids=self.target_ids[i],
                target_idxs=self.target_idxs[i],
                target_labels=self.target_labels[i],
                mask_idx=self.mask_idx[i],
            )
        else:
            return dict(
                input_ids=self.input_ids[i],
                labels=self.labels[i],
                target_ids=self.target_ids[i],
                target_idxs=self.target_idxs[i],
                hard_neg_ids=self.hard_neg_ids[i],
                target_labels=self.target_labels[i],
                hard_neg_labels=self.hard_neg_labels[i],
                mask_idx=self.mask_idx[i],
            )

@dataclass
class DataCollator(object):
    """Collate examples for supervised fine-tuning."""

    def __init__(
        self,
        ctx_tokenizer: transformers.PreTrainedTokenizer,
        question_tokenizer: transformers.PreTrainedTokenizer,
    ):
        self.ctx_tokenizer = ctx_tokenizer
        self.question_tokenizer = question_tokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        if "hard_neg_ids" in instances[0]:
            (
                input_ids,
                labels,
                target_ids,
                target_idxs,
                hard_neg_ids,
                target_labels,
                hard_neg_labels,
            ) = tuple(
                [instance[key] for instance in instances]
                for key in (
                    "input_ids",
                    "labels",
                    "target_ids",
                    "target_idxs",
                    "hard_neg_ids",
                    "target_labels",
                    "hard_neg_labels",
                )
            )
            hard_neg_ids = torch.nn.utils.rnn.pad_sequence(
                hard_neg_ids,
                batch_first=True,
                padding_value=self.ctx_tokenizer.pad_token_id,
            )
            batch_target_ids = []
            for target_id in target_ids:
                for item in target_id:
                    batch_target_ids.append(item)
            if batch_target_ids == []:
                target_ids = None
            else:
                target_ids = torch.nn.utils.rnn.pad_sequence(
                    batch_target_ids,
                    batch_first=True,
                    padding_value=self.ctx_tokenizer.pad_token_id,
                )
            labels = torch.nn.utils.rnn.pad_sequence(
                labels,
                batch_first=True,
                padding_value=self.question_tokenizer.pad_token_id,
            )
        else:
            if "target_ids" in instances[0]:
                (
                    input_ids,
                    labels,
                    target_ids,
                    target_idxs,
                    target_labels,
                ) = tuple(
                    [instance[key] for instance in instances]
                    for key in (
                        "input_ids",
                        "labels",
                        "target_ids",
                        "target_idxs",
                        "target_labels",
                    )
                )
                batch_target_ids = []
                for target_id in target_ids:
                    for item in target_id:
                        batch_target_ids.append(item)
                if batch_target_ids == []:
                    target_ids = None
                else:
                    target_ids = torch.nn.utils.rnn.pad_sequence(
                        batch_target_ids,
                        batch_first=True,
                        padding_value=self.ctx_tokenizer.pad_token_id,
                    )
                labels = torch.nn.utils.rnn.pad_sequence(
                    labels,
                    batch_first=True,
                    padding_value=self.question_tokenizer.pad_token_id,
                )
            else:
                if "labels" in instances[0]:
                    input_ids, labels, target_labels = tuple(
                        [instance[key] for instance in instances]
                        for key in (
                            "input_ids",
                            "labels",
                            "target_labels",
                        )
                    )
                    labels = torch.nn.utils.rnn.pad_sequence(
                        labels,
                        batch_first=True,
                        padding_value=self.question_tokenizer.pad_token_id,
                    )
                else:
                    input_ids, target_labels = tuple(
                        [instance[key] for instance in instances]
                        for key in ("input_ids", "target_labels")
                    )
                    labels = None
                target_ids = None
                target_idxs = None
            hard_neg_ids = None
            hard_neg_labels = None
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.question_tokenizer.pad_token_id,
        )
        alce_idx = (
            [instance["alce_idx"] for instance in instances]
            if "alce_idx" in instances[0]
            else None
        )
        mask_idx = (
            [instance["mask_idx"] for instance in instances]
            if "mask_idx" in instances[0]
            else None
        )
        return dict(
            input_ids=input_ids,
            labels=labels,
            target_ids=target_ids,
            hard_neg_ids=hard_neg_ids,
            target_labels=target_labels,
            target_idxs=target_idxs,
            mask_idx=mask_idx,
            attention_mask=input_ids.ne(self.question_tokenizer.pad_token_id),
            target_attention_mask=target_ids.ne(self.ctx_tokenizer.pad_token_id)
            if target_ids != None
            else None,
            hard_neg_mask=hard_neg_ids.ne(self.ctx_tokenizer.pad_token_id)
            if hard_neg_ids != None
            else None,
            hard_neg_labels=hard_neg_labels,
            alce_idx=alce_idx
        )

def train_data_module(
    ctx_tokenizer: transformers.PreTrainedTokenizer,
    question_tokenizer: transformers.PreTrainedTokenizer,
    dataset_config,
) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = TrainDataset(
        ctx_tokenizer=ctx_tokenizer,
        question_tokenizer=question_tokenizer,
        dataset_config=dataset_config,
        # dataset_type="train",
    )
    data_collator = DataCollator(
        ctx_tokenizer=ctx_tokenizer, question_tokenizer=question_tokenizer
    )
    return dict(
        train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator
    )

def test_data_module(
    ctx_tokenizer: transformers.PreTrainedTokenizer,
    question_tokenizer: transformers.PreTrainedTokenizer,
    dataset_config,
) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""

    if dataset_config.eval_docs == "":
        print(f"%% Working on Distractor Setting!")
        eval_dataset = ALCE_Top100_Dataset(
            ctx_tokenizer=ctx_tokenizer,
            question_tokenizer=question_tokenizer,
            dataset_config=dataset_config,
        )
        ctx_dataset = ALCE_Top100_EmbedDataset(
            ctx_tokenizer=ctx_tokenizer,
            question_tokenizer=question_tokenizer,
            dataset_config=dataset_config,
        )
    else:
        print(f"%% Working on Full Setting!")
        eval_dataset = ALCE_Top100_FullDataset(
            ctx_tokenizer=ctx_tokenizer,
            question_tokenizer=question_tokenizer,
            dataset_config=dataset_config,
        )
        ctx_dataset = ALCE_Top100_FullEmbedDataset(
            ctx_tokenizer=ctx_tokenizer,
            question_tokenizer=question_tokenizer,
            dataset_config=dataset_config,
        )
    data_collator = DataCollator(
        ctx_tokenizer=ctx_tokenizer, question_tokenizer=question_tokenizer
    )
    return dict(
        eval_dataset=eval_dataset, ctx_dataset=ctx_dataset, data_collator=data_collator
    )

