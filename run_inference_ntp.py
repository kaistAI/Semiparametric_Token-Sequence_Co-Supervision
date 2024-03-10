from tqdm import tqdm
import pickle
import copy
import sys
import os, re
import fire
import torch

from peft import prepare_model_for_int8_training

from transformers import (
    LlamaForCausalLM,
    LlamaModel,
    LlamaTokenizer,
    AutoTokenizer,
    AutoModel,
)

from configs import fsdp_config, training_config

from utils.config_utils import (
    update_config,
    generate_dataset_config,
)

from utils.inference_utils import (
    dump,
    test_ext_dpr_alce,
    test_ext_contriever_alce,
)

from dataset.generate_dataset import test_data_module
import json
from accelerate import Accelerator
from transformers import BertTokenizer, AutoModel


def main(**kwargs):
    if "training_argument" in kwargs.keys() and kwargs["training_argument"] is not None:
        with open(kwargs["training_argument"], "r") as f:
            json_obj = json.load(f)
        train_config = training_config(**json_obj)
    else:
        train_config = training_config()

    # Update the configuration for the training and sharding process
    update_config((train_config, fsdp_config), **kwargs)

    # Loading Model Checkpoints (If Checkpoint is in .distcp files, first convert into HF .bin files, and delete .distcp files)
    model_ckpt_path = os.path.join(
        train_config.dist_checkpoint_root_folder,
        train_config.dist_checkpoint_folder + "-" + train_config.model_name,
    )



    retriever_ckpt_path = os.path.join(
        train_config.dist_checkpoint_root_folder, train_config.ret_checkpoint_folder
    )
    torch.cuda.manual_seed(train_config.seed)
    torch.manual_seed(train_config.seed)

    dataset_config = generate_dataset_config(train_config, kwargs)
    update_config((dataset_config), **kwargs)

    if dataset_config.ndocs == -1:
        doc_path = dataset_config.eval_docs
        assert doc_path != ""
        doc_data = json.load(open(doc_path))
        dataset_config.ndocs = len(doc_data)

    print(f"Inference over Top {dataset_config.ndocs}")
    print(
        f"Working on .. {dataset_config.dataset} // batch size: {train_config.val_batch_size}"
    )
    print(f"Retrieval model: {train_config.retriever}")

    if train_config.retriever == "dpr":
        basepath = "sole_inference"
    elif train_config.retriever == "contriever":
        basepath = "sole_inference_contriever"
    elif train_config.retriever == "llama":
        basepath = "sole_inference_llama"
    else:
        assert False

    os.makedirs(f"{basepath}/{train_config.dist_checkpoint_folder}", exist_ok=True)
    """ 
    os.makedirs(f"../{basepath}/{train_config.dist_checkpoint_folder}", exist_ok=True)
    emb_path = f"../{basepath}/{train_config.dist_checkpoint_folder}/single_{train_config.single}.{dataset_config.dataset}.top{dataset_config.ndocs}.emb" 
    """ 
    os.makedirs(f"../{basepath}/{train_config.dist_checkpoint_folder}", exist_ok=True)
    emb_path = f"../{basepath}/{train_config.dist_checkpoint_folder}/single_{train_config.single}.{dataset_config.dataset}.top{dataset_config.ndocs}.emb" 
    #assert os.path.exists(emb_path)

    if train_config.ret_first:
        save_path = f"{basepath}/{train_config.dist_checkpoint_folder}/single_{train_config.single}.ret_first.{dataset_config.dataset}.top{dataset_config.ndocs}" 
        temp_path = f"{basepath}/{train_config.dist_checkpoint_folder}/single_{train_config.single}.ret_first.{dataset_config.dataset}.top{dataset_config.ndocs}.tmp" 
    else:
        save_path = f"{basepath}/{train_config.dist_checkpoint_folder}/single_{train_config.single}.{dataset_config.dataset}.top{dataset_config.ndocs}" 
        temp_path = f"{basepath}/{train_config.dist_checkpoint_folder}/single_{train_config.single}.{dataset_config.dataset}.top{dataset_config.ndocs}.tmp" 

    print(f"Emb path: {emb_path}\nsave_path: {save_path}\ntemp_path: {temp_path}")

    accelerator = Accelerator()
    world_size = int(accelerator.num_processes)

    if not os.path.exists(temp_path):
        # with accelerator.main_process_first():

        if os.path.exists(emb_path):
            print(f"Opening embedding from .. {emb_path}")
            with open(emb_path, "rb") as f:
                gathered_ctx = pickle.load(f)

            if train_config.retriever == "dpr":
                ctx_model, eval_tokenizer, ctx_tokenizer = prepare_dpr(
                    train_config, dataset_config, model_ckpt_path, "ctx", load=False, kwargs=kwargs,
                )
            elif train_config.retriever == "contriever":
                ctx_model, eval_tokenizer, ctx_tokenizer = prepare_contriever(
                    train_config, dataset_config, model_ckpt_path, "ctx", load=False, kwargs=kwargs
                )
            elif train_config.retriever == "llama":
                ctx_model, eval_tokenizer, ctx_tokenizer = prepare_llama(
                    train_config,
                    dataset_config,
                    model_ckpt_path,
                    retriever_ckpt_path,
                    "ctx",
                    load=False, 
                    kwargs=kwargs,
                )
            else:
                assert False

            data_module = test_data_module(
                ctx_tokenizer=ctx_tokenizer,
                question_tokenizer=eval_tokenizer,
                dataset_config=dataset_config,
            )
            ctx_dataset = data_module["ctx_dataset"]
            eval_dataset = data_module["eval_dataset"]
            data_collator_eval = data_module["data_collator"]
    
        else:
            print(f"Loading ctx encoder .. ")
            if train_config.retriever == "dpr":
                ctx_model, eval_tokenizer, ctx_tokenizer = prepare_dpr(
                    train_config, dataset_config, model_ckpt_path, "ctx", kwargs
                )
            elif train_config.retriever == "contriever":
                ctx_model, eval_tokenizer, ctx_tokenizer = prepare_contriever(
                    train_config, dataset_config, model_ckpt_path, "ctx", kwargs
                )
            elif train_config.retriever == "llama":
                ctx_model, eval_tokenizer, ctx_tokenizer = prepare_llama(
                    train_config,
                    dataset_config,
                    model_ckpt_path,
                    retriever_ckpt_path,
                    "ctx",
                    kwargs,
                )
            else:
                assert False

            data_module = test_data_module(
                ctx_tokenizer=ctx_tokenizer,
                question_tokenizer=eval_tokenizer,
                dataset_config=dataset_config,
            )
            ctx_dataset = data_module["ctx_dataset"]
            eval_dataset = data_module["eval_dataset"]
            data_collator_eval = data_module["data_collator"]

            dataloader = torch.utils.data.DataLoader(
                ctx_dataset,
                batch_size=16,  # train_config.val_batch_size,
                num_workers=train_config.num_workers_dataloader,
                pin_memory=True,
                sampler=None,
                drop_last=False,
                collate_fn=data_collator_eval,
            )
            ctx_model, dataloader = accelerator.prepare(ctx_model, dataloader)

            print(f"Start dumping embeddings ...")
            if train_config.retriever == "contriever":
                gathered_ctx = dump(
                    ctx_model,
                    train_config,
                    dataloader,
                    None,
                    world_size,
                    do_mean_pooling=True,
                )
            else:
                gathered_ctx = dump(
                    ctx_model,
                    train_config,
                    dataloader,
                    None,
                    world_size,
                    do_mean_pooling=False,
                )
            with open(emb_path, "wb") as f:
                pickle.dump(gathered_ctx, f)
            accelerator.wait_for_everyone()
            print(f"Done dumping embeddings in {emb_path}!")

            accelerator.free_memory()
            del dataloader, ctx_model

        # accelerator.wait_for_everyone()
        # print("Everyone is here")
        with accelerator.main_process_first():
            if train_config.retriever == "dpr":
                q_model, generation_model, eval_tokenizer = prepare_dpr(
                    train_config, dataset_config, model_ckpt_path, "question", kwargs
                )
            elif train_config.retriever == "contriever":
                q_model, generation_model, eval_tokenizer = prepare_contriever(
                    train_config, dataset_config, model_ckpt_path, "question", kwargs
                )
            elif train_config.retriever == "llama":
                q_model, generation_model, eval_tokenizer = prepare_llama(
                    train_config,
                    dataset_config,
                    model_ckpt_path,
                    retriever_ckpt_path,
                    "question",
                    kwargs,
                )
            else:
                assert False

        dataloader = torch.utils.data.DataLoader(
            eval_dataset,
            batch_size=train_config.val_batch_size,
            num_workers=train_config.num_workers_dataloader,
            pin_memory=True,
            sampler=None,
            drop_last=False,
            collate_fn=data_collator_eval,
        )

        q_model, generation_model, dataloader = accelerator.prepare(
            q_model, generation_model, dataloader
        )
        print("model loading done!")
        accelerator.wait_for_everyone()
        print("all model loading done!")

        if train_config.retriever in ["dpr", "llama"]:
            gathered_results, gathered_ret_idx = test_ext_dpr_alce(
                generation_model,
                q_model,
                train_config,
                dataset_config,
                dataloader,
                gathered_ctx,
                eval_tokenizer,
                ctx_tokenizer,
                None,
                world_size,
            )
        elif train_config.retriever == "contriever":
            gathered_results, gathered_ret_idx = test_ext_contriever_alce(
                generation_model,
                q_model,
                train_config,
                dataset_config,
                dataloader,
                gathered_ctx,
                eval_tokenizer,
                ctx_tokenizer,
                None,
                world_size,
            )
        else:
            assert False

        with open(temp_path, "wb") as f:
            pickle.dump(
                {
                    "gathered_results": gathered_results,
                    "gathered_ret_idx": gathered_ret_idx,
                },
                f,
            )

        print(f"Temp Saving in .. {temp_path}")
        #os.system(f"rm {emb_path}")

    else:
        print(f"Loading .. {temp_path}")
        with open(temp_path, "rb") as file:
            f = pickle.load(file)
        print(f"Done loading .. {temp_path}")
        gathered_results = f["gathered_results"]
        gathered_ret_idx = f["gathered_ret_idx"]

    if accelerator.is_local_main_process:
        input_path = dataset_config.eval_data
        if input_path.endswith(".json"):
            input_data = json.load(open(input_path))
        else:
            import jsonlines

            with jsonlines.open(input_path, "r") as jsonl_f:
                input_data = [obj for obj in jsonl_f]

        doc_path = dataset_config.eval_docs
        if doc_path == "":
            doc_data = []
        else:
            doc_data = json.load(open(doc_path))

        new_results = {"data": [], "args": [],
                    "total_cost": 0.0, "azure_filter_fail": ""}

        if not os.path.exists(temp_path) and os.path.exists(save_path):
            save_data = json.load(open(save_path))
            if "data" in save_data:
                if len(save_data['data']) == len(input_data):
                    print(f"Done!")
                    os.system(f"python eval_metric.py --f {save_path} --data_name {dataset_config.dataset.replace('alce_', '')}")
                    sys.exit(-1)
        save_data = {}

        for instance_idx in tqdm(range(len(input_data))):
            item = input_data[instance_idx]
            prompt = item["question"]

            if prompt in save_data:
                continue

            if len(doc_data) == 0:
                ctxs = item["docs"][: dataset_config.ndocs]
            else:
                ctxs = doc_data

            if type(list(gathered_results.keys())[0]) == str:
                instance_idx = str(instance_idx)
            text = gathered_results[instance_idx]
            inf_ret_idx = gathered_ret_idx[instance_idx]
            # print("ctxs: ", ctxs)
            # print("text: ", text)
            text = text.split("## Output:\n\n")[1]
            item["raw_output"] = copy.deepcopy(text)
            # print("post_text: ", text)

            doc_items = []
            for doc_item in ctxs:
                if doc_item["id"] in inf_ret_idx:
                    doc_items.append(doc_item)

            item["docs"] = doc_items

            if dataset_config.ctx_truncate:
                # split by [Ce]
                if "[Ce]" not in text and len(inf_ret_idx) != 0:
                    print("add [Ce] at the end!")
                    text = f"{text} [Ce]"
                text = text.replace("[Ret] [Ce]", "[Ret]")
                if len(inf_ret_idx) == 0:
                    text = text.replace("[Ce]", "")
                    text = text.replace("[Cs]", "")
                text = text.replace("</s>", "")

                if len(inf_ret_idx) == 0:
                    new_text = text
                else:
                    text_list = text.split("[Ce]")
                    # cs_cnt = 0
                    # ce_index = [m.start() for m in re.finditer('\[Ce\]', text)]
                    # assert len(ce_index) == len(inf_ret_idx), f"## text: {text}\n\n## inf_ret_idx: {inf_ret_idx}"

                    rev_text_list = []
                    cur_ret_idx = []
                    for ce_idx, ce_text in enumerate(text_list):
                        if ce_idx >= len(inf_ret_idx):
                            continue
                        elif "[Cs]" not in ce_text:
                            continue
                        else:
                            while ce_text.startswith(" "):
                                ce_text = ce_text[1:]
                            if ce_text == "":
                                continue

                            _cs_cnt = ce_text.count("[Cs]")
                            if _cs_cnt > 1:
                                print(f"++ Over 1:\n{ce_text}\n")

                            ret_idx = inf_ret_idx[ce_idx]
                            # except:
                            #     print(f"## text: {text}\n\nce_text: {ce_text}\n\n## inf_ret_idx: {inf_ret_idx}")
                            #     import sys; sys.exit(-1)
                            # _ret_idx += 1
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

                    new_text = " ".join(text_list)

                if instance_idx % 10 == 0:
                    print("-" * 80)
                    print(f"inf_ret_idx: {inf_ret_idx}")
                    print("-" * 80)
                    for elem in item["docs"]:
                        print(elem)
                        print("-" * 80)
                    print("=" * 80)
                    print("text: ", item["raw_output"])
                    print(f"new_text: {new_text}")
                # input()
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
                        if (
                            cs_idx >= len(cs_index)
                            or ret_index[ret_idx] >= cs_index[cs_idx]
                        ):
                            continue
                        while (
                            ce_idx < len(ce_index) - 1
                            and cs_index[cs_idx] >= ce_index[ce_idx]
                        ):
                            ce_idx += 1
                        if (
                            ce_idx >= len(ce_index)
                            or cs_index[cs_idx] >= ce_index[ce_idx]
                        ):
                            continue
                        cite_pair.append(
                            (ret_index[ret_idx], cs_index[cs_idx], ce_index[ce_idx])
                        )
                new_text = ""
                intermediate = []
                pin = 0
                for ret, cs, ce in cite_pair:
                    new_text += text[pin:ret]
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
                            new_text += f"[{citation_idx}]."
                        else:
                            new_text += cite_item
                            new_text += f"[{citation_idx}]"
                    # intermediate.append("[Retrieval]", cite_item)
                    pin = ce + 4
                new_text += text[pin:]

            # for doc_text in docs:
            #     flag = False
            #     for doc_item in ctxs:
            #         if doc_item["text"] in doc_text:
            #             flag = True
            #             doc_items.append(doc_item)
            #             break
            #     if flag ==False:
            #         assert False

            # item["intermediate"] = intermediate

            # print("new_text: ", new_text)
            # print("docs: ", docs)
            item["output"] = new_text
            save_data[prompt] = item

            with open(save_path, "w") as f:
                json.dump(save_data, f)

        new_results["data"] = list(save_data.values())
        with open(save_path, "w") as writer:
            json.dump(new_results, writer)
        print(f"**** Done! Saving in .. {save_path}")
        os.system(f"rm {temp_path}")

        os.system(f"python eval_metric.py --f {save_path} --data_name {dataset_config.dataset.replace('alce_', '')}")

def prepare_llama(
    train_config, dataset_config, model_ckpt_path, retriever_ckpt_path, type, load=True, kwargs=None
):
    # Load the tokenizer and add special tokens
    eval_tokenizer = LlamaTokenizer.from_pretrained(
        model_ckpt_path, model_max_length=512 # 1024
    )

    eval_tokenizer.add_special_tokens(dict(pad_token="<PAD>"))
    if train_config.natural_form:
        eval_tokenizer.add_special_tokens(
            {"additional_special_tokens": ["[Cs]", "[Ce]"]}
        )
        eval_tokenizer.add_special_tokens({"additional_special_tokens": ["[Ret]"]})
    # Load the pre-trained model and setup its configuration

    if type == "ctx":
        if train_config.single:
            print(f"[single] Loading Llama ctx encoder")
            model = LlamaModel.from_pretrained(
                # train_config.model_name,
                retriever_ckpt_path,
                # train_config,
                load_in_8bit=True if train_config.quantization else None,
                device_map="auto" if train_config.quantization else None,
            )
            model.resize_token_embeddings(len(eval_tokenizer))
        else:
            print(f"[dual] Loading Llama ctx encoder")
            model = LlamaModel.from_pretrained(
                # train_config.model_name,
                os.path.join(retriever_ckpt_path, "ctx"),
                # train_config,
                load_in_8bit=True if train_config.quantization else None,
                device_map="auto" if train_config.quantization else None,
            )
            model.resize_token_embeddings(len(eval_tokenizer))
            # model = model.ctx_encoder
    elif type == "question":
        if not train_config.single:
            print(f"[dual] Loading Llama Q encoder")
            q_model = LlamaModel.from_pretrained(
                # train_config.model_name,
                retriever_ckpt_path,
                # train_config,
                load_in_8bit=True if train_config.quantization else None,
                device_map="auto" if train_config.quantization else None,
            )
            q_model.resize_token_embeddings(len(eval_tokenizer))
        else:
            print(f"[single] Loading Llama q encoder")
            q_model = LlamaModel.from_pretrained(
                # train_config.model_name,
                retriever_ckpt_path,
                # train_config,
                load_in_8bit=True if train_config.quantization else None,
                device_map="auto" if train_config.quantization else None,
            )
            q_model.resize_token_embeddings(len(eval_tokenizer))
        # q_model = model.model
        print(f"Loading LlamaForCausalLM")
        model = LlamaForCausalLM.from_pretrained(
            # train_config.model_name,
            model_ckpt_path,
            load_in_8bit=True if train_config.quantization else None,
            device_map="auto" if train_config.quantization else None,
        )
    else:
        assert False

    print(f"Model embed token size => {model.get_input_embeddings().weight.shape}")

    # Prepare the model for int8 training if quantization is enabled
    if train_config.quantization:
        model = prepare_model_for_int8_training(model)

    if type == "ctx":
        return model, eval_tokenizer, eval_tokenizer
    else:
        return q_model, model, eval_tokenizer


def prepare_dpr(train_config, dataset_config, model_ckpt_path, type, kwargs):
    # Load the tokenizer and add special tokens
    eval_tokenizer = LlamaTokenizer.from_pretrained(
        model_ckpt_path, model_max_length=1024
    )

    eval_tokenizer.add_special_tokens(dict(pad_token="<PAD>"))
    if train_config.natural_form:
        eval_tokenizer.add_special_tokens(
            {"additional_special_tokens": ["[Cs]", "[Ce]"]}
        )
        eval_tokenizer.add_special_tokens({"additional_special_tokens": ["[Ret]"]})
    # Load the pre-trained model and setup its configuration

    if type == "ctx":
        ctx_tokenizer = BertTokenizer.from_pretrained(
            "sentence-transformers/facebook-dpr-question_encoder-single-nq-base"
        )

    if type == "ctx":
        print(f"Loading DPR ctx encoder")
        model = AutoModel.from_pretrained(
            "sentence-transformers/facebook-dpr-ctx_encoder-single-nq-base",
            load_in_8bit=True if train_config.quantization else None,
            device_map="auto" if train_config.quantization else None,
        )
    elif type == "question":
        print(f"Loading DPR Q encoder")
        q_model = AutoModel.from_pretrained(
            "sentence-transformers/facebook-dpr-question_encoder-single-nq-base",
            load_in_8bit=True if train_config.quantization else None,
            device_map="auto" if train_config.quantization else None,
        )
        print(f"Loading LlamaForCausalLM")
        model = LlamaForCausalLM.from_pretrained(
            # train_config.model_name,
            model_ckpt_path,
            load_in_8bit=True if train_config.quantization else None,
            device_map="auto" if train_config.quantization else None,
        )
    else:
        assert False

    print(f"Model embed token size => {model.get_input_embeddings().weight.shape}")

    # Prepare the model for int8 training if quantization is enabled
    if train_config.quantization:
        model = prepare_model_for_int8_training(model)

    # model.to(torch.bfloat16)
    if type == "ctx":
        return model, eval_tokenizer, ctx_tokenizer
    else:
        return q_model, model, eval_tokenizer


def prepare_contriever(train_config, dataset_config, model_ckpt_path, type, kwargs):
    # Load the tokenizer and add special tokens
    eval_tokenizer = LlamaTokenizer.from_pretrained(
        model_ckpt_path, model_max_length=1024
    )

    eval_tokenizer.add_special_tokens(dict(pad_token="<PAD>"))
    if train_config.natural_form:
        eval_tokenizer.add_special_tokens(
            {"additional_special_tokens": ["[Cs]", "[Ce]"]}
        )
        eval_tokenizer.add_special_tokens({"additional_special_tokens": ["[Ret]"]})
    # Load the pre-trained model and setup its configuration

    if type == "ctx":
        ctx_tokenizer = AutoTokenizer.from_pretrained("facebook/contriever-msmarco")

    if type == "ctx":
        print(f"Loading ctx contriever-msmarco")
        assert not train_config.quantization
        model = AutoModel.from_pretrained(
            "facebook/contriever-msmarco",
            load_in_8bit=True if train_config.quantization else None,
            device_map="auto" if train_config.quantization else None,
        )
    elif type == "question":
        print(f"Loading Q contriever-msmarco")
        assert not train_config.quantization
        q_model = AutoModel.from_pretrained(
            "facebook/contriever-msmarco",
            load_in_8bit=True if train_config.quantization else None,
            device_map="auto" if train_config.quantization else None,
        )
        print(f"Loading LlamaForCausalLM")
        model = LlamaForCausalLM.from_pretrained(
            # train_config.model_name,
            model_ckpt_path,
            load_in_8bit=True if train_config.quantization else None,
            device_map="auto" if train_config.quantization else None,
        )
    else:
        assert False

    print(f"Model embed token size => {model.get_input_embeddings().weight.shape}")

    # Prepare the model for int8 training if quantization is enabled
    if train_config.quantization:
        model = prepare_model_for_int8_training(model)

    # model.to(torch.bfloat16)
    if type == "ctx":
        return model, eval_tokenizer, ctx_tokenizer
    else:
        return q_model, model, eval_tokenizer


if __name__ == "__main__":
    fire.Fire(main)
