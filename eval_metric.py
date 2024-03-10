import argparse
import collections
import json
import re
import string
import torch
import copy

from nltk import sent_tokenize
import nltk

nltk.download("punkt")
import numpy as np
from rouge import Rouge
from rouge_score import rouge_scorer, scoring
from tqdm import tqdm
import sys
import logging

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline

from collections import defaultdict
from utils.alce_utils import normalize_answer, get_max_memory, remove_citations

QA_MODEL = "gaotianyu1350/roberta-large-squad"
AUTOAIS_MODEL = "google/t5_xxl_true_nli_mixture"

global autoais_model, autoais_tokenizer
autoais_model, autoais_tokenizer = None, None


from typing import Any, Dict, List, Tuple


class Metric:
    """
    An abstract class representing a metric which can be accumulated.
    """

    def __call__(self, predictions: Any, gold_labels: Any):
        raise NotImplementedError

    def get_metric(self, reset: bool) -> Dict[str, Any]:
        """
        Compute and return the metric. Optionally also call `self.reset`.
        """
        raise NotImplementedError

    def reset(self) -> None:
        """
        Reset any accumulators or internal state.
        """
        raise NotImplementedError


def compute_f1(a_gold, a_pred):
    """Compute F1 score between two strings."""

    def _get_tokens(s):
        if not s:
            return []
        return normalize_answer(s).split()

    gold_toks = _get_tokens(a_gold)
    pred_toks = _get_tokens(a_pred)

    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())

    if len(gold_toks) == 0 or len(pred_toks) == 0:
        # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
        return int(gold_toks == pred_toks)

    if num_same == 0:
        return 0

    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)

    return f1


def compute_exact(a_gold, a_pred):
    """Check whether two strings are equal up to normalization."""

    return int(normalize_answer(a_gold) == normalize_answer(a_pred))


def exact_presence(short_answers, context):
    """Verify if any of the answers is present in the given context.
    Args:
        short_answers: list of short answers to look for in the context
        context: a paragraph to search for short answers
    Returns:
        true if any of the short answers is present in the context
    """
    if type(short_answers) == str:
        short_answers = [short_answers]
    n_short_answers = [normalize_answer(sa) for sa in short_answers]
    n_context = normalize_answer(context)

    for ans in n_short_answers:
        if ans in n_context:
            # print(f"Successfully found .. answer {ans} in context: {n_context}")
            return True

    # print(f"Failed to find .. answer {ans} in context: {n_context}")
    return False


def _change_fever_ans(ans):
    if ans == "SUPPORTS":
        return "true"
    elif ans == "REFUTES":
        return "false"
    else:
        assert False
        return None


"""
metric for wow 
"""


def compute_unigram_f1(data, support_prec_list):
    score = []
    ret_success_score = []
    for item, prec in zip(data, support_prec_list):
        assert len(item["answers"]) == 1, f"answers:: {item['answers']}"
        answer = normalize_answer(item["answers"][0])
        output = normalize_answer(item["output"])
        gold_toks = answer.split()
        pred_toks = output.split()

        # calculate_f1
        common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
        num_same = sum(common.values())

        if len(gold_toks) == 0 or len(pred_toks) == 0:
            # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
            # return int(gold_toks == pred_toks)
            score.append(int(gold_toks == pred_toks))

        if num_same == 0:
            score.append(0)
            ret_success_score.append(0)
        else:
            precision = 1.0 * num_same / len(pred_toks)
            recall = 1.0 * num_same / len(gold_toks)
            f1 = (2 * precision * recall) / (precision + recall)
            score.append(f1)
            if prec == 0:
                ret_success_score.append(0)
            else:
                ret_success_score.append(f1)

    score = np.array(score).mean() * 100
    ret_success_score = np.array(ret_success_score).mean() * 100
    return score, ret_success_score


def compute_qa_single(data, data_name, support_prec_list, is_raw, ctx_truncate):
    score = []
    success_score = []
    fail_score = []
    etc_score = []
    if data_name == "nq":
        assert len(data) == len(support_prec_list)
    for item, prec in zip(data, support_prec_list):
        if is_raw:
            if ctx_truncate:
                docs = [f"{el['title']} :: {el['text']}" for el in item["docs"]]
                output = " ".join(docs)
                output = f"{output} {item['raw_output']}"
            else:
                output = item["raw_output"]
        else:
            output = item["output"]

        existence = exact_presence(item["answers"], output)
        if existence:
            if prec > 0:
                success_score.append(100)
                etc_score.append(100)
            else:
                fail_score.append(100)
                etc_score.append(0)
            score.append(100)
        else:
            if prec > 0:
                success_score.append(0)
            else:
                fail_score.append(0)
            score.append(0)
            etc_score.append(0)
        item["answer_em"] = existence
        item["ret_prec"] = prec
        # print(f"existence: {existence} || prec: {prec}")
        # input('Press Enter to Continue .. ') 
    assert len(score) == len(success_score) + len(fail_score)
    check_score = np.array(success_score + fail_score).mean()
    score = np.array(score).mean()
    assert score == check_score
    success_score = np.array(success_score).mean()
    fail_score = np.array(fail_score).mean()
    etc_score = np.array(etc_score).mean()
    normalized_score = -1
    n_success_score = -1
    n_fail_score = -1

    if data_name == "fever":
        normalized_score = []
        n_success_score = []
        n_fail_score = []
        for item, prec in zip(data, support_prec_list):
            answers = [_change_fever_ans(elem) for elem in item["answers"]] + item[
                "answers"
            ]
            existence = exact_presence(answers, item["output"])
            if existence:
                if prec > 0:
                    n_success_score.append(100)
                else:
                    n_fail_score.append(100)
                normalized_score.append(100)
            else:
                if prec > 0:
                    n_success_score.append(0)
                else:
                    n_fail_score.append(0)
                normalized_score.append(0)
        normalized_score = np.array(normalized_score).mean()
        n_success_score = np.array(n_success_score).mean()
        n_fail_score = np.array(n_fail_score).mean()

    etc = {
        "success_score": success_score,
        "fail_score": fail_score,
        "n_success_score": n_success_score,
        "n_fail_score": n_fail_score,
        "only_ret": etc_score
    }
    return score, data, normalized_score, etc


def compute_rouge(data):
    """Main function for rouge scoring.
    If two references are provided,
    the best score is chosen for each instance.
    Args:
        data: requires field `output` and `answer` (or `annotations` for ASQA)
        metrics: list of evaluation metrics
    Returns:
        dictionary representation of rouge scores
    """

    def _rouge_calculation(
        hypotheses, references1, references2=[], metrics=["rougeLsum"]
    ):
        if references2 == []:
            references2 = references1

        scorer = rouge_scorer.RougeScorer(metrics, use_stemmer=True)
        aggregator = scoring.BootstrapAggregator()

        for i in range(len(hypotheses)):
            scores1 = scorer.score(references1[i], hypotheses[i])
            scores2 = scorer.score(references2[i], hypotheses[i])
            if scores1["rougeLsum"].fmeasure > scores2["rougeLsum"].fmeasure:
                aggregator.add_scores(scores1)
            else:
                aggregator.add_scores(scores2)

        scores = {m: [] for m in metrics}

        for m in metrics:
            fmeasure = aggregator.aggregate()[m].mid.fmeasure
            scores[m].append(fmeasure)

        for m in scores:
            scores[m] = 100 * sum(scores[m]) / len(scores[m])

        return scores

    hypotheses = {}
    references1 = {}
    references2 = {}

    for idx, item in enumerate(data):
        hypotheses[idx] = item["output"]
        if "annotations" in item and item["annotations"] is not None:  # For ASQA
            references1[idx] = item["annotations"][0]["long_answer"]
            references2[idx] = item["annotations"][1]["long_answer"]
        else:
            references1[idx] = item["answer"]
            references2[idx] = item["answer"]

    h, r1, r2 = [], [], []

    for key in references1:
        h.append(hypotheses[key])
        r1.append(references1[key])

        if references2 is not None:
            r2.append(references2[key])

    h = ["\n".join(sent_tokenize(text.lower())) for text in h]
    r1 = ["\n".join(sent_tokenize(text.lower())) for text in r1]
    r2 = ["\n".join(sent_tokenize(text.lower())) for text in r2]
    scores = _rouge_calculation(h, r1, r2)

    return scores["rougeLsum"]


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


class AnswerMetric(Metric):
    def __init__(self) -> None:
        self._total_em = 0.0
        self._total_f1 = 0.0
        self._count = 0
        self._total_em_list = []
        self._total_f1_list = []

    def __call__(
        self,
        predicted_answer: str,
        ground_truth_answers: List[str],
    ):
        exact_scores = metric_max_over_ground_truths(
            compute_exact, predicted_answer, ground_truth_answers
        )
        f1_scores = metric_max_over_ground_truths(
            compute_f1, predicted_answer, ground_truth_answers
        )

        self._total_em += int(exact_scores)
        self._total_em_list.append(int(exact_scores))
        self._total_f1 += f1_scores
        self._total_f1_list.append(f1_scores)
        self._count += 1

    def get_metric(self, reset: bool = False) -> Tuple[float, float]:
        exact_match = self._total_em / self._count if self._count > 0 else 0
        f1_score = self._total_f1 / self._count if self._count > 0 else 0
        if reset:
            self.reset()
        return exact_match, f1_score, self._total_em_list, self._total_f1_list

    def reset(self):
        self._total_em = 0.0
        self._total_f1 = 0.0
        self._total_em_list = []
        self._total_f1_list = []
        self._count = 0


def compute_str_em(data):
    """Compute STR-EM metric (only for ASQA)
    Args:
        data: requires field `qa_pairs/short_answers` and `output`
    Returns:
        STR-EM and STR-EM-HIT ()
    """
    acc = []
    hit = []

    if "qa_pairs" not in data[0] or data[0]["qa_pairs"] is None:
        return 0, 0

    for item in data:
        loc_acc = []
        for qa_pair in item["qa_pairs"]:
            loc_acc.append(exact_presence(qa_pair["short_answers"], item["output"]))
        acc.append(np.mean(loc_acc))
        hit.append(int(np.mean(loc_acc) == 1))

    return 100 * np.mean(acc), 100 * np.mean(hit)


def compute_ans_em(data):
    acc = []
    wo_acc = []
    for item in data:
        score = exact_presence([str(item["answer"])], item["output"])
        acc.append(score)
        if str(item["answer"]) in ["yes", "no", "true", "false"]:
            continue
        else:
            wo_acc.append(score)
    print(f"# without yes/no/true/false: {len(wo_acc)}/{len(acc)}")
    return 100 * np.mean(acc), 100 * np.mean(wo_acc)


def compute_qa_musique(data):
    answer_metric = AnswerMetric()
    for item in data:
        answer_metric(item["output"], [item["answer"]] + item["answer_alias"])
    exact_match, f1_score, total_em, total_f1 = answer_metric.get_metric()
    f1 = round(f1_score * 100, 3)
    em = round(exact_match * 100, 3)
    for elem, _em, _f1 in zip(data, total_em, total_f1):
        elem["em"] = _em
        elem["f1"] = _f1
    return f1, em, data


def compute_str_f1(data):
    f1_scores = []
    for item in data:
        all_answers = [item["answer"]] + item["answer_aliases"]
        f1_scores.append(
            max([compute_f1(answer, item["output"]) for answer in all_answers])
        )
    return 100 * np.mean(f1_scores)


def compute_len(data):
    """Compute average length of predictions."""

    res, cntr = 0, 0
    for item in data:
        res += len(item["output"].split())
        cntr += 1
    return res / cntr


def compute_qa(data, data_name):
    """Compute QA-based accuracy.
    Args:
        data: requires filed `qa_pairs/short_answers` and `output`
    Returns:
        QA metrics (QA-EM, QA-F1, QA-Hit)
    """

    # Load model
    logger.info("Loading the RoBERTa-large SQuAD model for QA-based accuracy...")
    qa_pipeline = pipeline("question-answering", model=QA_MODEL, device=0)
    logger.info("Done")

    if data_name in ["hotpotqa", "musique", "strategyqa"]:
        for item in tqdm(data):
            question = [item["question"]]
            context = item["output"] if len(item["output"]) > 0 else " "
            results = qa_pipeline(
                question=question, context=context, handle_impossible_answer=True
            )
            loc_counter, loc_em, loc_f1 = 0, 0, 0

            for idx, res in enumerate(results):
                answer = item["answer"]
                prediction = res["answer"]

                loc_em += max([compute_exact(a, prediction) for a in answers])
                loc_f1 += max([compute_f1(a, prediction) for a in answers])
                loc_counter += 1

            em.append(loc_em / loc_counter)
            f1.append(loc_f1 / loc_counter)
            bins.append(loc_em == loc_counter)

    else:
        if "qa_pairs" not in data[0] or data[0]["qa_pairs"] is None:
            logger.warn("Warning: no QA pairs found in data")
            return {
                "QA-EM": 0,
                "QA-F1": 0,
                "QA-Hit": 0,
            }

        # Get prediction
        logger.info("Computing the QA-based accuracy...")
        em, f1, bins = [], [], []
        for item in tqdm(data):
            question = [qa_pair["question"] for qa_pair in item["qa_pairs"]]
            context = item["output"] if len(item["output"]) > 0 else " "
            results = qa_pipeline(
                question=question, context=context, handle_impossible_answer=True
            )
            loc_counter, loc_em, loc_f1 = 0, 0, 0

            for idx, res in enumerate(results):
                answers = item["qa_pairs"][idx]["short_answers"]
                prediction = res["answer"]

                loc_em += max([compute_exact(a, prediction) for a in answers])
                loc_f1 += max([compute_f1(a, prediction) for a in answers])
                loc_counter += 1

            em.append(loc_em / loc_counter)
            f1.append(loc_f1 / loc_counter)
            bins.append(loc_em == loc_counter)

    return {
        "QA-EM": 100 * np.mean(em),
        "QA-F1": 100 * np.mean(f1),
        "QA-Hit": 100 * np.mean(bins),
    }


def compute_mauve(data):
    """Compute Mauve score."""

    logger.info("Computing MAUVE...")
    human_data = []
    model_data = []
    for item in data:
        # Remove ending punctuations
        # Remove any new lines
        # Truncate by 100 words
        human_data.append(
            " ".join(
                (item["question"] + " " + item["answer"].strip()).split()[:100]
            ).rstrip(string.punctuation)
        )
        model_data.append(
            " ".join(
                (item["question"] + " " + item["output"].strip()).split()[:100]
            ).rstrip(string.punctuation)
        )

    import mauve

    out = mauve.compute_mauve(
        p_text=human_data,
        q_text=model_data,
        device_id=0,
        max_text_length=512,
        verbose=True,
        batch_size=8,
        featurize_model_name="gpt2-large",
    )
    return out.mauve * 100


def _run_nli_autoais(passage, claim):
    """
    Run inference for assessing AIS between a premise and hypothesis.
    Adapted from https://github.com/google-research-datasets/Attributed-QA/blob/main/evaluation.py
    """
    global autoais_model, autoais_tokenizer
    input_text = "premise: {} hypothesis: {}".format(passage, claim)
    input_ids = autoais_tokenizer(input_text, return_tensors="pt", max_length=1024, truncation=True).input_ids.to(autoais_model.device)
    with torch.inference_mode():
        outputs = autoais_model.generate(input_ids, max_new_tokens=10)
    result = autoais_tokenizer.decode(outputs[0], skip_special_tokens=True)
    inference = 1 if result == "1" else 0
    return inference


def compute_fever_nli(data, support_prec_list):
    global autoais_model, autoais_tokenizer
    if autoais_model is None:
        logger.info("Loading AutoAIS model...")
        autoais_model = AutoModelForSeq2SeqLM.from_pretrained(
            AUTOAIS_MODEL,
            torch_dtype=torch.bfloat16,
            max_memory=get_max_memory(),
            device_map="auto",
        )
        autoais_tokenizer = AutoTokenizer.from_pretrained(AUTOAIS_MODEL, use_fast=False)

    logger.info("Computing claims...")
    scores = []
    ret_success_scores = []
    for item, prec in zip(data, support_prec_list):
        # passage: f["data"][0]["output"]
        # claims: f["data"][0]["question"]
        normalized_output = remove_citations(item["output"])
        claims = item["question"]
        entail = _run_nli_autoais(normalized_output, claims)
        ans = item["answers"]
        assert len(set(ans)) == 1, f"ans: {ans}"
        # entail == 1 ->
        ans = ans[0]

        if ans == "SUPPORTS":
            if entail == 1:
                scores.append(100)
                if prec == 0:
                    ret_success_scores.append(0)
                else:
                    ret_success_scores.append(100)
            else:
                scores.append(0)
                ret_success_scores.append(0)
        elif ans == "REFUTES":
            if entail == 1:
                scores.append(0)
                ret_success_scores.append(0)
            else:
                scores.append(100)
                if prec == 0:
                    ret_success_scores.append(0)
                else:
                    ret_success_scores.append(100)
        else:
            assert False, f"ans: {ans}"
    
    return np.mean(scores), np.mean(ret_success_scores) 

def compute_claims(data):
    global autoais_model, autoais_tokenizer
    if autoais_model is None:
        logger.info("Loading AutoAIS model...")
        autoais_model = AutoModelForSeq2SeqLM.from_pretrained(
            AUTOAIS_MODEL,
            torch_dtype=torch.bfloat16,
            max_memory=get_max_memory(),
            device_map="auto",
        )
        autoais_tokenizer = AutoTokenizer.from_pretrained(AUTOAIS_MODEL, use_fast=False)

    logger.info("Computing claims...")
    scores = []
    for item in tqdm(data):
        normalized_output = remove_citations(item["output"])
        entail = 0
        claims = item["claims"]
        for claim in claims:
            entail += _run_nli_autoais(normalized_output, claim)
        scores.append(entail / len(claims))
    return 100 * np.mean(scores)

def parse_cs_ce(text, docs, ctx_truncate):
    if ctx_truncate:
        # assert False
        # print(f"Input text: {text}\n")
        text_list = text.split("[Ce]")
        # print(f"text_list: {text_list}\n")
        n_text_list = []
        cs_cnt = 0

        inf_ret_idx = [elem["id"] for elem in docs] 
        rev_text_list = []
        cur_ret_idx = []
        for ce_idx, ce_text in enumerate(text_list):
            if "[Cs]" not in ce_text:
                continue
            _cs_cnt = ce_text.count("[Cs]")
            assert _cs_cnt == 1
            for _ in range(_cs_cnt):
                ce_text = ce_text.split("[Cs]")[-1]
                ret_idx = inf_ret_idx[cs_cnt]
                if ret_idx in cur_ret_idx:
                    citation_idx = cur_ret_idx.index(ret_idx) + 1
                else:
                    cur_ret_idx.append(ret_idx)
                    citation_idx = len(cur_ret_idx)

                if ce_text[-1] == ".":
                    ce_text = ce_text[:-1] + f"[{citation_idx}]. "
                else:
                    ce_text += f"[{citation_idx}] "

            ce_text = ce_text.replace("[Cs]", "")
            ce_text = ce_text.replace("[Ce]", "")
            while ce_text.startswith(" "):
                ce_text = ce_text[1:]
            while ce_text.endswith(" "):
                ce_text = ce_text[:-1]
            text_list[ce_idx] = ce_text
            n_text_list.append(ce_text)

        new_text = " ".join(n_text_list)
        # print(f"New text: {new_text}\n")
        # print("="*80)
        # print('\n\n')

    else:
        ret_index = [m.start() for m in re.finditer("\[Ret\]", text)]
        cs_index = [m.start() for m in re.finditer("\[Cs\]", text)]
        ce_index = [m.start() for m in re.finditer("\[Ce\]", text)]
        cs_idx = 0
        ce_idx = 0
        cite_pair = []
        docs = []
        if len(ret_index) > 0 and len(cs_index) > 0 and len(ce_index) > 0:
            for ret_idx in range(len(ret_index)):
                if len(cite_pair) > 0:
                    if (
                        ret_index[ret_idx] < cite_pair[-1][1]
                        or ret_index[ret_idx] < cite_pair[-1][2]
                    ):
                        continue
                while (
                    cs_idx < len(cs_index) - 1
                    and ret_index[ret_idx] >= cs_index[cs_idx]
                ):
                    cs_idx += 1
                if ret_index[ret_idx] >= cs_index[cs_idx]:
                    continue
                while (
                    ce_idx < len(ce_index) - 1
                    and cs_index[cs_idx] >= ce_index[ce_idx]
                ):
                    ce_idx += 1
                if cs_index[cs_idx] >= ce_index[ce_idx]:
                    continue
                cite_pair.append(
                    (ret_index[ret_idx], cs_index[cs_idx], ce_index[ce_idx])
                )

        new_text = ""
        intermediate = []
        pin = 0
        for ret, cs, ce in cite_pair:
            document = text[ret + 5 : cs].strip()
            cite_item = text[cs + 4 : ce].strip()

            if document in docs:
                citation_idx = docs.index(document) + 1
            else:
                docs.append(document)
                citation_idx = len(docs)

            if len(cite_item) > 0:
                if cite_item[-1] == ".":
                    new_text += cite_item[:-1]
                    new_text += f"[{citation_idx}]. "
                else:
                    new_text += cite_item
                    new_text += f"[{citation_idx}] "
            # intermediate.append("[Retrieval]", cite_item)
            pin = ce + 4
        new_text = new_text.replace("</s>", "")
    return new_text

def revised_compute_autoais(
    data,
    doc_file_path,
    decontext=False,
    concat=False,
    qampari=False,
    at_most_citations=None,
    ctx_truncate=False
):
    """
    Compute AutoAIS score.

    Args:
        data: requires field `output` and `docs`
              - docs should be a list of items with fields `title` and `text` (or `phrase` and `sent` for QA-extracted docs)
        citation: check citations and use the corresponding references.
        decontext: decontextualize the output
    """

    global autoais_model, autoais_tokenizer
    if autoais_model is None:
        logger.info("Loading AutoAIS model...")
        autoais_model = AutoModelForSeq2SeqLM.from_pretrained(
            AUTOAIS_MODEL,
            torch_dtype=torch.bfloat16,
            max_memory=get_max_memory(),
            device_map="auto",
        )
        autoais_tokenizer = AutoTokenizer.from_pretrained(AUTOAIS_MODEL, use_fast=False)

    logger.info(f"Running AutoAIS...")

    if doc_file_path is not None:
        all_docs = json.load(doc_file_path)
        candidate_docs = dict()
        for doc in all_docs:
            candidate_docs[doc.pop("id")] = doc

    def _format_document(doc):
        """Format document for AutoAIS."""

        if "sent" in doc:
            # QA-extracted docs
            return "Title: %s\n%s" % (doc["title"], doc["sent"])
        else:
            if "title" in doc.keys():
                return "Title: %s\n%s" % (doc["title"], doc["text"])
            else:
                return doc["text"]

    ais_scores = []
    rev_ais_scores = []
    ais_scores_prec = []

    ret_ais_scores = []
    ret_rev_ais_scores = []
    ret_ais_scores_prec = []

    wrong_ais_scores = []
    wrong_rev_ais_scores = []
    wrong_ais_scores_prec = []

    sent_total = 0
    sent_mcite = 0
    sent_mcite_support = 0
    sent_mcite_overcite = 0
    autoais_log = []
    for item in tqdm(data):
        if "rev.ref_prec" in item:
            prec_score = item["rev.ref_prec"]
        elif "ref_prec" in item:
            prec_score = item["ref_prec"]
        else:
            prec_score = 100
        if doc_file_path is None:
            candidate_docs = item["docs"]

        # Get sentences by using NLTK
        if qampari:
            _sents = [
                item["question"] + " " + x.strip()
                for x in item["output"].rstrip().rstrip(".").rstrip(",").split(",")
            ]
        else:
            # print(f"** Raw output: {item['raw_output']}")
            sen = parse_cs_ce(item["raw_output"], item["docs"], ctx_truncate=ctx_truncate)
            # print(f"** After parsing cs-ce: {sen}")
            _sents = sent_tokenize(sen)
            # print(f"** After tokenizing: {_sents}")

        if len(_sents) == 0:
            # print(f"Pass.. len(sents) == 0")
            continue

        # print("="*80)
        # print(f"raw output: {item['raw_output']}\n")

        _target_sents = [remove_citations(sent).strip() for sent in _sents]

        entail = 0
        entail_prec = 0
        total_citations = 0
        skip_sen = 0
        no_citation_sen = 0

        # Iterate over each sentences and combine the ones with out references
        assert len(_target_sents) == len(_sents)
        target_sents = []; sents = []; idx_list = []
        for i in range(len(_sents)):
            # find references
            ref = [
                int(r[1:]) - 1 for r in re.findall(r"\[\d+", _sents[i])
            ] 
            idx_list.append(i)
            if len(ref) == 0:
                continue 
            else:
                sents.append(" ".join([_sents[sid] for sid in idx_list]))
                target_sents.append(" ".join([_target_sents[sid] for sid in idx_list]))
                idx_list = []

        # print(f"After combining by cs-ce: {sents}\n")

        # iterate over each sentences
        for sent_id, sent in enumerate(sents):
            # print(f"## <sent_id: {sent_id}> {sent}")
            target_sent = target_sents[
                sent_id
            ]  # Citation removed and (if opted for) decontextualized
            joint_entail = -1  # Undecided

            # Find references
            ref = [
                int(r[1:]) - 1 for r in re.findall(r"\[\d+", sent)
            ]  # In text citation id starts from 1
            #logger.info(f"For `{sent}`, find citations {ref}")
            if len(ref) == 0:
                # No citations
                joint_entail = 0
                skip_sen += 1
                joint_entail = 0
            elif any([ref_id >= len(candidate_docs) for ref_id in ref]):
                # Citations out of range
                joint_entail = 0
            else:
                if at_most_citations is not None:
                    ref = ref[:at_most_citations]
                total_citations += len(ref)
                joint_passage = "\n".join(
                    [_format_document(candidate_docs[psgs_id]) for psgs_id in ref]
                )

            # If not directly rejected by citation format error, calculate the recall score
            if joint_entail == -1:
                t_sen_list = sent_tokenize(target_sent)
                _joint_entail = 0
                for t_sen in t_sen_list: 
                    _joint_entail += _run_nli_autoais(joint_passage, t_sen)
                    print(f"\n\n## [RECALL] Autoais over .. \npassage: {joint_passage}\nsent: {t_sen}\n-> Score: {_joint_entail}\n\n")
                if len(t_sen_list) == 0:
                    print(f"raw_output: {item['raw_output']}\ntarget_sent: {target_sent}\nt_sen_list: {t_sen_list}")
                    joint_entail = 0
                else:
                    joint_entail = _joint_entail / len(t_sen_list)
                # joint_entail = _run_nli_autoais(joint_passage, target_sent)
                autoais_log.append(
                    {
                        "question": item["question"],
                        "output": item["output"],
                        "claim": sent,
                        "passage": [joint_passage],
                        "model_type": "NLI",
                        "model_output": joint_entail,
                    }
                )

            entail += joint_entail
            if len(ref) > 1:
                sent_mcite += 1

            # calculate the precision score if applicable
            if joint_entail and len(ref) > 1:
                sent_mcite_support += 1
                # Precision check: did the model cite any unnecessary documents?
                for psgs_id in ref:
                    # condition A
                    passage = _format_document(candidate_docs[psgs_id])
                    nli_result = _run_nli_autoais(passage, target_sent)

                    # condition B
                    if not nli_result:
                        subset_exclude = copy.deepcopy(ref)
                        subset_exclude.remove(psgs_id)
                        passage = "\n".join(
                            [
                                _format_document(candidate_docs[pid])
                                for pid in subset_exclude
                            ]
                        )
                        nli_result = _run_nli_autoais(passage, target_sent)
                        if nli_result:  # psgs_id is not necessary
                            flag = 0
                            sent_mcite_overcite += 1
                        else:
                            entail_prec += 1
                    else:
                        entail_prec += 1
            else:
                entail_prec += joint_entail

        sent_total += len(sents)

        if len(sents) == skip_sen:
            # no_citation += 1
            rev_ais_scores.append(0)
            ret_rev_ais_scores.append(0)
        elif len(sents) > skip_sen:
            rev_ais_scores.append(entail / (len(sents) - skip_sen))
            if prec_score > 0:
                ret_rev_ais_scores.append(entail / (len(sents) - skip_sen))
            else:
                ret_rev_ais_scores.append(0)
        else:
            assert False

        rec = entail / len(sents)
        prec =  entail_prec / total_citations if total_citations > 0 else 0
        ais_scores.append(rec)
        ais_scores_prec.append(prec)  # len(sents))
        # print(f"## precision: {ais_scores_prec[-1]} || recall: {ais_scores[-1]}")

        if prec_score > 0:
            ret_ais_scores.append(rec)
            ret_ais_scores_prec.append(prec)
            wrong_ais_scores.append(0)
            wrong_ais_scores_prec.append(0)
        else:
            ret_ais_scores.append(0)
            ret_ais_scores_prec.append(0)
            wrong_ais_scores.append(rec)
            wrong_ais_scores_prec.append(prec)


    if sent_mcite > 0 and sent_mcite_support > 0:
        print(
            "Among all sentences, %.2f%% have multiple citations, among which %.2f%% are supported by the joint set, among which %.2f%% overcite."
            % (
                100 * sent_mcite / sent_total,
                100 * sent_mcite_support / sent_mcite,
                100 * sent_mcite_overcite / sent_mcite_support,
            )
        )

    print(f"citation_rec.revised: {100 * np.mean(rev_ais_scores)}\ncitation_rec: {100 * np.mean(ais_scores)}\ncitation_prec: {100 * np.mean(ais_scores_prec)}")
    return {
        "rev.citation_rec": 100 * np.mean(ais_scores),
        "ret.rev.citation_rec": 100 * np.mean(ret_ais_scores),
        "wrong.rev.citation_rec": 100 * np.mean(wrong_ais_scores),
    }


def compute_autoais(
    data,
    doc_file_path,
    decontext=False,
    concat=False,
    qampari=False,
    at_most_citations=None,
):
    """
    Compute AutoAIS score.

    Args:
        data: requires field `output` and `docs`
              - docs should be a list of items with fields `title` and `text` (or `phrase` and `sent` for QA-extracted docs)
        citation: check citations and use the corresponding references.
        decontext: decontextualize the output
    """

    global autoais_model, autoais_tokenizer
    if autoais_model is None:
        logger.info("Loading AutoAIS model...")
        autoais_model = AutoModelForSeq2SeqLM.from_pretrained(
            AUTOAIS_MODEL,
            torch_dtype=torch.bfloat16,
            max_memory=get_max_memory(),
            device_map="auto",
        )
        autoais_tokenizer = AutoTokenizer.from_pretrained(AUTOAIS_MODEL, use_fast=False)

    logger.info(f"Running AutoAIS...")

    if doc_file_path is not None:
        all_docs = json.load(doc_file_path)
        candidate_docs = dict()
        for doc in all_docs:
            candidate_docs[doc.pop("id")] = doc

    def _format_document(doc):
        """Format document for AutoAIS."""

        if "sent" in doc:
            # QA-extracted docs
            return "Title: %s\n%s" % (doc["title"], doc["sent"])
        else:
            if "title" in doc.keys():
                return "Title: %s\n%s" % (doc["title"], doc["text"])
            else:
                return doc["text"]

    ais_scores = []
    rev_ais_scores = []
    ais_scores_prec = []

    ret_ais_scores = []
    ret_rev_ais_scores = []
    ret_ais_scores_prec = []

    wrong_ais_scores = []
    wrong_rev_ais_scores = []
    wrong_ais_scores_prec = []

    sent_total = 0
    sent_mcite = 0
    sent_mcite_support = 0
    sent_mcite_overcite = 0
    autoais_log = []
    for item in tqdm(data):
        if "rev.ref_prec" in item:
            prec_score = item["rev.ref_prec"]
        else:
            prec_score = item["ref_prec"]
        if doc_file_path is None:
            if "docs" not in item:
               candidate_docs = []
            else:
               candidate_docs = item["docs"]

        # Get sentences by using NLTK
        if qampari:
            sents = [
                item["question"] + " " + x.strip()
                for x in item["output"].rstrip().rstrip(".").rstrip(",").split(",")
            ]
        else:
            sents = sent_tokenize(item["output"])

        if len(sents) == 0:
            continue

        target_sents = [remove_citations(sent).strip() for sent in sents]

        entail = 0
        entail_prec = 0
        total_citations = 0
        skip_sen = 0
        no_citation_sen = 0

        # iterate over each sentences
        for sent_id, sent in enumerate(sents):
            target_sent = target_sents[
                sent_id
            ]  # Citation removed and (if opted for) decontextualized
            joint_entail = -1  # Undecided

            # Find references
            ref = [
                int(r[1:]) - 1 for r in re.findall(r"\[\d+", sent)
            ]  # In text citation id starts from 1
            #logger.info(f"For `{sent}`, find citations {ref}")
            if len(ref) == 0:
                # No citations
                joint_entail = 0
                skip_sen += 1
                joint_entail = 0
            elif any([ref_id >= len(candidate_docs) for ref_id in ref]):
                # Citations out of range
                joint_entail = 0
            else:
                if at_most_citations is not None:
                    ref = ref[:at_most_citations]
                total_citations += len(ref)
                joint_passage = "\n".join(
                    [_format_document(candidate_docs[psgs_id]) for psgs_id in ref]
                )

            # If not directly rejected by citation format error, calculate the recall score
            if joint_entail == -1:
                joint_entail = _run_nli_autoais(joint_passage, target_sent)
                autoais_log.append(
                    {
                        "question": item["question"],
                        "output": item["output"],
                        "claim": sent,
                        "passage": [joint_passage],
                        "model_type": "NLI",
                        "model_output": joint_entail,
                    }
                )

            entail += joint_entail
            if len(ref) > 1:
                sent_mcite += 1

            # calculate the precision score if applicable
            if joint_entail and len(ref) > 1:
                sent_mcite_support += 1
                # Precision check: did the model cite any unnecessary documents?
                for psgs_id in ref:
                    # condition A
                    passage = _format_document(candidate_docs[psgs_id])
                    nli_result = _run_nli_autoais(passage, target_sent)

                    # condition B
                    if not nli_result:
                        subset_exclude = copy.deepcopy(ref)
                        subset_exclude.remove(psgs_id)
                        passage = "\n".join(
                            [
                                _format_document(candidate_docs[pid])
                                for pid in subset_exclude
                            ]
                        )
                        nli_result = _run_nli_autoais(passage, target_sent)
                        if nli_result:  # psgs_id is not necessary
                            flag = 0
                            sent_mcite_overcite += 1
                        else:
                            entail_prec += 1
                    else:
                        entail_prec += 1
            else:
                entail_prec += joint_entail

        sent_total += len(sents)

        if len(sents) == skip_sen:
            # no_citation += 1
            rev_ais_scores.append(0)
            ret_rev_ais_scores.append(0)
        elif len(sents) > skip_sen:
            rev_ais_scores.append(entail / (len(sents) - skip_sen))
            if prec_score > 0:
                ret_rev_ais_scores.append(entail / (len(sents) - skip_sen))
            else:
                ret_rev_ais_scores.append(0)
        else:
            assert False

        rec = entail / len(sents)
        prec = entail_prec / total_citations if total_citations > 0 else 0
        ais_scores.append(rec)
        ais_scores_prec.append(prec)  # len(sents))
        if prec_score > 0:
            ret_ais_scores.append(rec)
            ret_ais_scores_prec.append(prec)
            wrong_ais_scores.append(0)
            wrong_ais_scores_prec.append(0)
        else:
            ret_ais_scores.append(0)
            ret_ais_scores_prec.append(0)
            wrong_ais_scores.append(rec)
            wrong_ais_scores_prec.append(prec)

    if sent_mcite > 0 and sent_mcite_support > 0:
        print(
            "Among all sentences, %.2f%% have multiple citations, among which %.2f%% are supported by the joint set, among which %.2f%% overcite."
            % (
                100 * sent_mcite / sent_total,
                100 * sent_mcite_support / sent_mcite,
                100 * sent_mcite_overcite / sent_mcite_support,
            )
        )

    return {
        "citation_rec": 100 * np.mean(ais_scores),
        "ret.citation_rec": 100 * np.mean(ret_ais_scores),
        "wrong.citation_rec": 100 * np.mean(wrong_ais_scores),
    }

def rougel_score(data, support_prec_list):
    score_list = []
    ret_success_score_list = []
    for item, prec in zip(data, support_prec_list):
        _score_list = []
        _ret_success_score_list = []
        prediction = item["output"]
        gt = item["answers"]
        for ans in gt:
            _score = _rougel_score(prediction, ans)
            _score_list.append(_score)
            if prec == 0:
                _ret_success_score_list.append(0)
            else:
                _ret_success_score_list.append(_score)
        score_list.append(max(_score_list))
        ret_success_score_list.append(max(_ret_success_score_list))
    return np.array(score_list).mean()*100, np.array(ret_success_score_list).mean()*100

def _rougel_score(prediction, ground_truth):
    rouge = Rouge()
    # no normalization
    try:
        scores = rouge.get_scores(prediction, ground_truth, avg=True)
    except ValueError:  # "Hypothesis is empty."
        return 0.0
    return scores["rouge-l"]["f"]

def compute_qampari_f1(data, cot=False):
    prec = []
    rec = []
    rec_top5 = []
    f1 = []
    f1_top5 = []

    num_preds = []
    for item in data:
        if cot:
            if ":" in item["output"]:
                o = ":".join(
                    item["output"].split(":")[1:]
                )  # try to separate the COT part and the answer list part.
            else:
                o = ""
        else:
            o = item["output"]
        preds = [
            normalize_answer(x.strip())
            for x in o.rstrip().rstrip(".").rstrip(",").split(",")
        ]
        preds = [p for p in preds if len(p) > 0]  # delete empty answers
        num_preds.append(len(preds))
        answers = [[normalize_answer(x) for x in ans] for ans in item["answers"]]
        flat_answers = [item for sublist in answers for item in sublist]

        prec.append(
            sum([p in flat_answers for p in preds]) / len(preds)
            if len(preds) > 0
            else 0
        )
        rec.append(sum([any([x in preds for x in a]) for a in answers]) / len(answers))
        rec_top5.append(
            min(5, sum([any([x in preds for x in a]) for a in answers]))
            / min(5, len(answers))
        )
        if (prec[-1] + rec[-1]) == 0:
            f1.append(0)
        else:
            f1.append(2 * prec[-1] * rec[-1] / (prec[-1] + rec[-1]))
        if (prec[-1] + rec_top5[-1]) == 0:
            f1_top5.append(0)
        else:
            f1_top5.append(2 * prec[-1] * rec_top5[-1] / (prec[-1] + rec_top5[-1]))

    return {
        "num_preds": np.mean(num_preds),
        "qampari_prec": 100 * np.mean(prec),
        "qampari_rec": 100 * np.mean(rec),
        "qampari_rec_top5": 100 * np.mean(rec_top5),
        "qampari_f1": 100 * np.mean(f1),
        "qampari_f1_top5": 100 * np.mean(f1_top5),
    }

def revised_compute_ref_support(data):
    ret_range = defaultdict(int)
    ref_em_list = []
    ref_prec_list = []
    empty = 0
    for idx, item in enumerate(data):
        doc_id = []
        output_refs = []
        if "docs" not in item:
            item["docs"] = []
        for elem in item["docs"]:
            if elem["id"] in doc_id:
                continue 
            else:
                doc_id.append(elem["id"])
                output_refs.append(f"{elem['title']} :: {elem['text']}")
        
        ret_range[len(output_refs)] += 1
        if type(item["answers"]) == str:
            answers = [item["answers"]]
        else:
            answers = item["answers"]

        if len(answers) == 0:
            empty += 1
            ref_prec_list.append(0)
            item["rev.ref_prec"] = 0
            continue

        if len(output_refs) == 0:
            ref_em = 0
            prec = 0

        else:
            existence = []
            for text in output_refs:
                _exist = False
                for ans in answers:
                    if ans in text:
                        _exist = True
                        break 
                existence.append(_exist)
            
            if sum(existence) > 0:
                ref_em = 100
            else:
                ref_em = 0

            prec = sum(existence) / len(output_refs) * 100 

        ref_em_list.append(ref_em)
        ref_prec_list.append(prec)
        item["rev.ref_em"] = ref_em
        item["rev.ref_prec"] = prec 

    ref_em = np.mean(ref_em_list)
    ref_prec = np.mean(ref_prec_list) 
    return ref_em, data, ref_prec, ref_prec_list


def compute_ref_support(data):
    ref_f1 = []
    ref_em = []
    ref_exist = []
    ret_range = defaultdict(int)
    ref_recall_list = []
    ref_prec_list = []
    empty = 0
    for idx, item in enumerate(data):
        if "docs" not in item:
            output_refs = []
        else:
            output_refs = [elem["id"] for elem in item["docs"]]
        output_refs = list(set(output_refs))
        ret_range[len(output_refs)] += 1

        if len(item["gold_docs"]) == 0:
            empty += 1
            ref_prec_list.append(0)
            ref_recall_list.append(0)
            item["ref_prec"] = 0
            continue
            # print("here")
            # sys.exit(-1)

        if type(item["gold_docs"][0]) == dict:
            item["gold_docs"] = set([el["id"] for el in item["gold_docs"]])

        prec_count, rec_count = 0, 0
        for ref_id in output_refs:
            if ref_id in item["gold_docs"]:
                prec_count += 1
        for doc_id in item["gold_docs"]:
            if doc_id in output_refs:
                rec_count += 1

        if len(output_refs) == 0:
            ref_precision = 0
        else:
            ref_precision = prec_count / len(output_refs)
        ref_prec_list.append(ref_precision)
        assert len(item["gold_docs"]) != 0
        ref_recall = rec_count / len(item["gold_docs"])
        ref_recall_list.append(ref_recall)

        if ref_precision + ref_recall == 0:
            _ref_f1 = 0
        else:
            _ref_f1 = (2 * ref_precision * ref_recall) / (ref_precision + ref_recall)
        _ref_em = set(output_refs) == set(item["gold_docs"])
        if prec_count == 0:
            _ref_exist = 0
        else:
            _ref_exist = 1

        item["ref_em"] = _ref_em
        item["ref_exist"] = _ref_exist
        item["ref_f1"] = _ref_f1
        item["ref_recall"] = ref_recall
        item["ref_prec"] = ref_precision

        # print(f"output: {item['output']}")
        # print(f"pred: {output_refs}")
        # print(f"gold: {item['gold_docs']}")
        # print(f"f1: {_ref_f1} || em: {_ref_em}")
        # print("="*80)
        # input()

        ref_f1.append(_ref_f1)
        ref_em.append(_ref_em)
        ref_exist.append(_ref_exist)

    ref_f1 = np.mean(ref_f1) * 100
    ref_em = np.mean(ref_em) * 100
    ref_exist = np.mean(ref_exist) * 100
    ref_recall = np.mean(ref_recall_list) * 100
    ref_prec = np.mean(ref_prec_list) * 100
    return ref_f1, ref_em, ref_exist, ret_range, data, ref_recall, ref_prec, empty, ref_prec_list


def parse_truncate(raw_output):
    """
    "raw_output": " [Cs] <PAD> Scott Derrickson is American.  [Ce] [Cs] Scott Derrickson is American.  [Ce] [Cs] Scott Derrickson is a film director from United States.  [Ce] Scott Derrickson is an American film director, screenwriter, producer, and author.  [Cs] Scott Derrickson is a director, screenwriter, producer, and author.  [Ce] Scott Derrickson is a film director from United States. Scott Derrickson is an American film director, screenwriter, producer, and author.</s>",
    "output": "[Cs] <PAD> Scott Derrickson is American.  [Ce] [Cs] Scott Derrickson is American.  [Ce] [Cs] Scott Derrickson is a film director from United States.  [Ce] Scott Derrickson is an American film director, screenwriter, producer, and author.  [Cs] Scott Derrickson is a director, screenwriter, producer, and author.  [Ce] Scott Derrickson is a film director from United States. Scott Derrickson is an American film director, screenwriter, producer, and author.</s>",
    """


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_name",
        type=str,
        default=None,
        choices=[
            "alce_hotpotqa",
            "alce_musique",
            "alce_strategyqa",
            "alce_asqa",
            "alce_eli5",
            "alce_qampari",
            "kilt_nq",
            "kilt_hotpotqa",
            "kilt_wow",
            "kilt_fever",
            "kilt_zsre",
            "kilt_trex",
            "kilt_triviaqa",
            "kilt_eli5",
        ],
        help="Eval Data name",
        required=True,
    )
    parser.add_argument(
        "--hotpotqa_version",
        type=str,
        default="distractor",
        choices=["distractor", "full"],
        help="HotpotQA Data Version",
    )

    parser.add_argument(
        "--doc_file",
        type=str,
        default=None,
        help="Path to Full Corpus(Documents). Used only when each instance does not have field 'docs'",
    )
    parser.add_argument(
        "--f",
        type=str,
        required=True,
        help="Output file. Should have field `question`, `output`, (ROUGE) `answer`, \
                        (accuracy) `qa_pairs`, (AIS) `docs`",
    )

    parser.add_argument("--rouge", action="store_true", help="Evaluate ROUGE score")
    parser.add_argument("--qa", action="store_true", help="Use the QA model")
    parser.add_argument(
        "--mauve", action="store_true", help="Use the mauve score model"
    )
    parser.add_argument(
        "--citations", action="store_true", help="Evaluation with citation"
    )
    parser.add_argument(
        "--at_most_citations",
        type=int,
        default=3,
        help="At most take this many documents (mostly for precision)",
    )
    parser.add_argument(
        "--retrieval_perf", action="store_true", help="Calculate retrieval performance"
    )
    parser.add_argument("--claims_nli", action="store_true", help="Use claims for ELI5")
    parser.add_argument("--no_save", action="store_true", help="Don't save  the result")
    parser.add_argument("--no_citation", action="store_true", help="Don't save  the result")
    parser.add_argument("--remove_before_ret", action="store_true", help="Remove those before Ret")
    parser.add_argument("--cal_by_raw", action="store_true", help="calculate with the ones before parsing (raw output)")
    parser.add_argument("--ctx_truncate", action="store_true", help="When trained with ctx_truncate. Necessary for parsing.")

    # QAMPARI
    parser.add_argument(
        "--cot",
        action="store_true",
        help="For QAMPARI, try to find colon and separate the COT and answer listing",
    )

    args = parser.parse_args()

    with open(args.f) as f:
        data_with_config = json.load(f)
    if "data" in data_with_config:
        data = data_with_config["data"]
    else:
        data = data_with_config

    if "qampari" in args.f:
        args.rouge = False
        args.qa = False
        args.mauve = False
        args.decontext = False
        qampari = True
    else:
        qampari = False


    if "oracle" in args.f or "wrong" in args.f:
        for elem in data:
            try:
                elem["gold_docs"] = elem["docs"]
            except:
                print(f"docs: {elem['docs']}")
                import sys; sys.exit(-1)

    # Truncate by newline and remove on the fly search result
    logger.warning(
        "We remove all the pre/appended space/newlines and we truncate the answer by the first newline."
    )
    logger.warning(
        "We replace any on the fly search result to standard bracket citation format."
    )
    for i in range(len(data)):
        if i<5: print(f"[1] output: {data[i]['output']}")
        if args.cal_by_raw:
            data[i]["output"] = data[i]["raw_output"]
        elif args.remove_before_ret:
            before_ret = data[i]["raw_output"].split("[Ret]")[0]
            if len(before_ret.replace(" ", "")) > 0:
                data[i]["output"] = data[i]["output"].replace(before_ret, "") 
        # data[i]['output'] = data[i]['output'].strip().split("\n")[0]
        data[i]['output'] = data[i]['output'].replace("\n", " ")
        data[i]['output'] = data[i]['output'].replace("<|im_end|>", "")
        data[i]['output'] = data[i]['output'].replace("<PAD>", "")
        data[i]['output'] = data[i]['output'].replace("</s>", "")
        data[i]['output'] = data[i]['output'].replace("[RELEVANT]", "")

        if i<5: 
            print(f"[1] output: {data[i]['output']}")
            print("-"*80)

    # Remove all citations for all non-AutoAIS evaluation
    normalized_data = copy.deepcopy(data)
    for i in range(len(normalized_data)):
        normalized_data[i]["output"] = remove_citations(normalized_data[i]["output"])

    result = {}
    ret_range = None
    result["length"] = compute_len(normalized_data)



    if "kilt" in args.data_name:
        
        if "nq" in args.data_name or "triviaqa" in args.data_name or "zsre" in args.data_name or "trex" in args.data_name:
            (
                result["rev.support_em"],
                data,
                result["rev.support_prec"],
                support_prec_list
            ) = revised_compute_ref_support(data)
            print(f"[RET] em: {result['rev.support_em']} || prec: {result['rev.support_prec']}") 
        else:
            (
                result["support_f1"],
                result["support_em"],
                result["support_exist"],
                ret_range,
                data,
                result["support_recall"],
                result["support_prec"],
                empty,
                support_prec_list,
            ) = compute_ref_support(data) 
    else:
        pass

    if args.data_name == "alce_musique":
        result["answer_f1"], result["answer_em"], normalized_data = compute_qa_musique(
            normalized_data
        )
    elif "kilt" in args.data_name:
        # etc = {"success_score": success_score, "fail_score":  fail_score, "n_success_score": n_success_score,  "n_fail_score": n_fail_score}
        result["answer_em"], normalized_data, normalized_score, etc = compute_qa_single(
            normalized_data, args.data_name, support_prec_list, is_raw=False, ctx_truncate=args.ctx_truncate
        )
        result["remove_ret_wrong_ans_right_em"] = etc["only_ret"]
        # print(result)
        # import sys; sys.exit(-1)
        if "nq" in args.data_name or "triviaqa" in args.data_name or "zsre" in args.data_name or "trex" in args.data_name:
            result["ret_success.answer_em"] = etc["success_score"]
            result["ret_fail.answer_em"] = etc["fail_score"]
            result["remove_ret_wrong_ans_right"] = etc["only_ret"]
            print(f"em: {result['answer_em']} || remove_ret_wrong: {etc['only_ret']}")
        if "fever" in args.data_name:
            result["answer_generous"] = normalized_score
            result["ret_success.answer_generous"] = etc["n_success_score"]
            result["ret_fail.answer_generous"] = etc["n_fail_score"]
            result["ans_nli"], result["remove_ret_wrong_ans_nli"] = compute_fever_nli(normalized_data, support_prec_list)
        if "wow" in args.data_name:
            result["unigram_f1"], result["remove_ret_wrong_unigram_f1"]  = compute_unigram_f1(normalized_data, support_prec_list)
        if "eli5" in args.data_name:
            result["rougel"], result["remove_ret_wrong_rougel"] = rougel_score(normalized_data, support_prec_list)

        if "raw_output" in data[0].keys():
           result["raw.answer_em"], normalized_data, normalized_score, etc = compute_qa_single(
               normalized_data, args.data_name, support_prec_list, is_raw=True, ctx_truncate=args.ctx_truncate
           )
           result["raw.remove_ret_wrong_ans_right_em"] = etc["only_ret"]
           if "nq" in args.data_name or "triviaqa" in args.data_name or "zsre" in args.data_name or "trex" in args.data_name:
               result["raw.ret_success.answer_em"] = etc["success_score"]
               result["raw.ret_fail.answer_em"] = etc["fail_score"]
               result["raw.remove_ret_wrong_ans_right"] = etc["only_ret"]
               print(f"[RAW] em: {result['raw.answer_em']} || remove_ret_wrong: {etc['only_ret']}")
           if "fever" in args.data_name:
               result["raw.answer_generous"] = normalized_score
               result["raw.ret_success.answer_generous"] = etc["n_success_score"]
               result["raw.ret_fail.answer_generous"] = etc["n_fail_score"]
               result["raw.ans_nli"], result["raw.remove_ret_wrong_ans_nli"] = compute_fever_nli(normalized_data, support_prec_list)
           if "wow" in args.data_name:
               result["raw.unigram_f1"], result["raw.remove_ret_wrong_unigram_f1"] = compute_unigram_f1(normalized_data, support_prec_list)
           if "eli5" in args.data_name:
               result["raw.rougel"], result["raw.remove_ret_wrong_rougel"]  = rougel_score(normalized_data, support_prec_list)

    elif args.data_name in ["alce_hotpotqa", "alce_strategyqa"]:
        result["answer_em"], result["answer_em_wo_bool"] = compute_ans_em(
            normalized_data
        )
    else:
        result["str_em"], result["str_hit"] = compute_str_em(normalized_data)

    if qampari:
        result.update(compute_qampari_f1(normalized_data, cot=args.cot))
    if args.rouge:
        result["rougeLsum"] = compute_rouge(normalized_data)
    if args.qa:
        result.update(compute_qa(normalized_data, args.data_name))
    if args.mauve:
        result["mauve"] = compute_mauve(normalized_data)
    if args.data_name == "alce_eli5":
        result["claims_nli"] = compute_claims(normalized_data)

    if not args.no_citation:
        if "vanilla" in args.f or "gtr" in args.f:
            result.update(
                compute_autoais(
                    data,
                    args.doc_file,
                    qampari=qampari,
                    at_most_citations=args.at_most_citations,
                )
            )
        else: 
            print(f"Running .. revised_compute_autoais")
            result.update(revised_compute_autoais(
                data,
                args.doc_file,
                qampari=qampari,
                at_most_citations=args.at_most_citations,
                ctx_truncate=args.ctx_truncate
            ))


    if ret_range is not None:
        save_dict = {"result": result, "retrieval_range": ret_range}
    else:
        save_dict = result

    print(save_dict)
    #print(f"No gold citation: {empty}/{len(data)}")

    if not args.no_save:
        if args.cal_by_raw:
            json.dump(save_dict, open(args.f + ".cal_by_raw.score", "w"), indent=4)
            json.dump(normalized_data, open(args.f + ".normalized_data.cal_by_raw.score.json", "w"), indent=4)
        elif args.remove_before_ret:
            json.dump(save_dict, open(args.f + ".remove_before_ret.score", "w"), indent=4)
            json.dump(normalized_data, open(args.f + ".normalized_data.remove_before_ret.score.json", "w"), indent=4)
        else:
            json.dump(save_dict, open(args.f + ".score", "w"), indent=4)
            json.dump(normalized_data, open(args.f + ".normalized_data.score.json", "w"), indent=4)
        # json.dump(data, open(args.f + ".data.score.json", "w"), indent=4)


if __name__ == "__main__":
    main()
