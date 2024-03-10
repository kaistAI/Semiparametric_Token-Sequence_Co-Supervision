# coding=utf-8
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" PyTorch LLaMA model."""
import re
from typing import List, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
import torch.nn.functional as F
from transformers.activations import ACT2FN
from transformers.modeling_outputs import CausalLMOutputWithPast
# from utils.modeling_utils import PreTrainedModel
from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from transformers import (
    LlamaForCausalLM,
    LlamaModel
)
from torch.distributed import get_rank

logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "LlamaConfig"


class JointModel(LlamaForCausalLM):
    def __init__(self, config, train_config):
        super().__init__(config)

        self.vocab_size = config.vocab_size
        self.config = config
        self.train_config = train_config
        self.model = LlamaModel(config)
        if not train_config.single:
            self.ctx_encoder = LlamaModel(config)
        else:
            self.ctx_encoder = self.model
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.all_gather = train_config.all_gather
        # Initialize weights and apply final processing
        self.post_init()

    def get_nb_trainable_parameters(self):
        r"""
        Returns the number of trainable parameters and number of all parameters in the model.
        """
        trainable_params = 0
        all_param = 0
        for _, param in self.named_parameters():
            num_params = param.numel()
            # if using DS Zero 3 and the weights are initialized empty
            if num_params == 0 and hasattr(param, "ds_numel"):
                num_params = param.ds_numel

            # Due to the design of 4bit linear layers from bitsandbytes
            # one needs to multiply the number of parameters by 2 to get
            # the correct number of parameters
            if param.__class__.__name__ == "Params4bit":
                num_params = num_params * 2

            all_param += num_params
            if param.requires_grad:
                trainable_params += num_params

        return trainable_params, all_param
    
    def print_trainable_parameters(self):
        """
        Prints the number of trainable parameters in the model.
        """
        trainable_params, all_param = self.get_nb_trainable_parameters()

        print(
            f"trainable params: {trainable_params:,d} || all params: {all_param:,d} || trainable%: {100 * trainable_params / all_param}"
        )
        
    def set_vanilla_vocab_size(self, vanilla_vocab_size):
        self.resize_token_embeddings(vanilla_vocab_size)
        self.ctx_encoder.resize_token_embeddings(vanilla_vocab_size)
        self.vanilla_vocab_size = vanilla_vocab_size

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def set_ids2text(self, ids2text):
        self.ids2text = ids2text

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def get_sId_list(self, sen):
        sId_list = re.findall(r"\[k_[0-9]+\]", sen)
        return sId_list

    def set_enc_embeddings(
        self, target_ids, hard_neg_ids, target_attention_mask, hard_neg_mask
    ):
        # if hard_negs is not None:
        #     target_ids.extend(self.remove_empty_in_hardneg(hard_negs))
        # assert "" not in input_texts
        # print("target_id: ", target_ids)
        # print("target_attn_mask_size: ", target_attention_mask)
        if target_ids is not None:
            embs = self.ctx_encoder(
                target_ids.to(self.ctx_encoder.device),
                attention_mask=target_attention_mask.to(self.ctx_encoder.device),
            ).last_hidden_state[:, -1, :].contiguous()
            embs = embs.to(self.model.device)
            assert embs.shape[-1] == self.config.hidden_size
        else:
            embs = None
        print(f"{torch.cuda.current_device()} embs  size: {embs.shape}", )

        if hard_neg_ids is not None:
            hard_neg_embs = self.ctx_encoder(
                hard_neg_ids.to(self.ctx_encoder.device),
                attention_mask=hard_neg_mask.to(self.ctx_encoder.device),
            ).last_hidden_state[:, -1, :].contiguous()
            hard_neg_embs = hard_neg_embs.to(self.model.device)
            assert hard_neg_embs.shape[-1] == self.config.hidden_size
            # print("hardneg embs  size: ", hard_neg_embs.shape)
        else:
            hard_neg_embs = None

        return embs, hard_neg_embs

    def remove_empty_in_hardneg(self, hardneg_list):
        ret_list = []
        for elem in hardneg_list:
            if elem == "":
                continue
            else:
                ret_list.append(elem)
        return ret_list

    def all_gather_set_enc_embeddings(
        self, target_ids, hard_neg_ids, target_attention_mask, hard_neg_mask
    ):
        if target_ids is not None:
            # print(f"[{torch.cuda.current_device()}] target_idx is not None")
            embs = (
                self.ctx_encoder(
                    target_ids.to(self.ctx_encoder.device),
                    attention_mask=target_attention_mask.to(self.ctx_encoder.device),
                )
                .last_hidden_state[:, -1, :]
                .contiguous()
            )
            # print(f"[{torch.cuda.current_device()}] embs.shape: {embs.shape}")
            assert embs.shape[-1] == self.config.hidden_size
            size = torch.LongTensor([embs.shape[0]]).to(self.ctx_encoder.device)
            # print(f"[{torch.cuda.current_device()}] size: {size}")
        else:
            # print(f"[{torch.cuda.current_device()}] target_idx is None")
            embs = (
                self.ctx_encoder(
                    torch.zeros([1, 100], dtype=torch.int64).to(self.ctx_encoder.device)
                )
                .last_hidden_state[:, -1, :]
                .contiguous()
            )
            # print(f"[{torch.cuda.current_device()}] embs.shape: {embs.shape}")
            size = torch.zeros(1, dtype=torch.int64).to(self.ctx_encoder.device)
            # print(f"[{torch.cuda.current_device()}] size: {size}")
        
        gathered_length = [
            torch.zeros_like(size).to(self.ctx_encoder.device)
            for _ in range(torch.distributed.get_world_size())
        ]
        torch.distributed.all_gather(gathered_length, size)
        # print(torch.cuda.current_device(), gathered_length)
        gathered_length_int = []
        for item in gathered_length:
            gathered_length_int.append(int(item[0].item()))
        # print(f"[{torch.cuda.current_device()}] gathered_length_int: {gathered_length_int}")

        if sum(gathered_length_int) == 0:
            assert False
            all_gathered_embs = None
        else:
            max_size = max(gathered_length_int) # max num from world size
            local_size = int(size[0].item())
            size_diff = max_size - local_size
            if size_diff:
                padding = torch.zeros(
                    [size_diff, self.config.hidden_size],
                    device=self.ctx_encoder.device,
                    dtype=embs.dtype,
                )
                padded_embs = torch.cat([embs, padding], dim=0)
            else:
                padded_embs = embs
            gathered_embs = [
                torch.zeros_like(padded_embs)
                for _ in range(torch.distributed.get_world_size())
            ]
            torch.distributed.all_gather(gathered_embs, padded_embs)
            gather_embs = []
            for rank, (item, length) in enumerate(
                zip(gathered_embs, gathered_length_int)
            ):
                # print(f"item:{item} length: {length}")
                # print(f"current rank: {get_rank()},ctx_encoder_device: {self.ctx_encoder.device}")
                if length != 0 and rank != get_rank():
                    gather_embs.append(
                        item[:length].detach().to(self.ctx_encoder.device)
                    )
            all_gathered_embs = torch.cat(gather_embs, dim=0)

        all_gathered_hardneg_embs = None
        if target_ids is not None:
            return embs, all_gathered_embs, all_gathered_hardneg_embs
        else:
            return None, all_gathered_embs, all_gathered_hardneg_embs

    def all_gather_convert_ids_global2local(
        self, input_ids, target_ids, hard_neg_ids, target_attention_mask, hard_neg_mask
    ):
        # print(f"[{torch.cuda.current_device()}] inside all_gather_convert_ids_global2local")
        embs, in_batch_embs, hard_neg_embs = self.all_gather_set_enc_embeddings(
            target_ids, hard_neg_ids, target_attention_mask, hard_neg_mask
        )
        concat_embs = []
        for item in [embs, in_batch_embs, hard_neg_embs]:
            if item is not None:
                concat_embs.append(item)
        if len(concat_embs) != 0:
            all_embs = torch.cat(concat_embs, dim=0)
        else:
            all_embs = None

        # if embs is not None:
        #     print(f"[{torch.cuda.current_device()}] Shape of all_embs: {all_embs.shape} || Shape of embs: {embs.shape} || Shape of in_batch_embs: {in_batch_embs.shape}")
        # else:
        #     print(f"[{torch.cuda.current_device()}] Shape of all_embs: {all_embs.shape} || Shape of embs: {embs} || Shape of in_batch_embs: {in_batch_embs.shape}")

        return input_ids, all_embs

    def extract_target_embs(self, input_ids, target_idxs, last_hidden_states):
        target_emb_list = []
        batch_size = input_ids.shape[0]
        assert batch_size == len(target_idxs)
        for i in range(batch_size):
            # do_print(input_ids[i], target_idxs[i])
            # for target_idx in target_idxs[i]:
            #     do_print(input_ids[i][target_idx])
            for target_idx in target_idxs[i]:
                if input_ids[i][target_idx] != self.vanilla_vocab_size - 1:
                    assert False
                target = last_hidden_states[i, target_idx, :]
                target_emb_list.append(target)

        target_embs = torch.stack(target_emb_list)

        return target_embs

    """
    train step => convert global id to local id
    """

    def convert_ids_global2local(
        self, input_ids, target_ids, hard_neg_ids, target_attention_mask, hard_neg_mask
    ):
        embs, hard_neg_embs = self.set_enc_embeddings(
            target_ids, hard_neg_ids, target_attention_mask, hard_neg_mask
        )
        concat_embs = []
        for item in [embs, hard_neg_embs]:
            if item is not None:
                concat_embs.append(item)
        if len(concat_embs) != 0:
            all_embs = torch.cat(concat_embs, dim=0)
        else:
            all_embs = None

        return input_ids, all_embs

    @replace_return_docstrings(
        output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC
    )
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        target_ids: Optional[torch.Tensor] = None,
        target_idxs: Optional[List] = None,
        hard_neg_ids: Optional[torch.Tensor] = None,
        target_labels: Optional[List[str]] = None,
        hard_neg_labels: Optional[List[str]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        target_attention_mask: Optional[torch.Tensor] = None,
        hard_neg_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, LlamaForCausalLM

        >>> model = LlamaForCausalLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
        >>> tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_TOKENIZER)

        >>> prompt = "Hey, are you consciours? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you consciours? Can you talk to me?\nI'm not consciours, but I can talk to you."
        ```"""
        # print("Ret input embedding: ", self.get_input_embeddings()(torch.LongTensor([32003]).to(self.ctx_encoder.device)))
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # decoder outputs consists of (dec_features, layer_state, decå_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = outputs[0]  # .last_hidden_state[:,-1,:]
        lm_logits = self.lm_head(hidden_states)
        logits = lm_logits
        loss = None
        # print("whole labels:", labels)
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = CrossEntropyLoss()  # (weight)

            shift_logits = shift_logits.view(
                -1, shift_logits.shape[-1]
            )  ## [BS*seq_len, vocab_size]
            shift_labels = shift_labels.view(-1)  ## [BS*seq_len]
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)

            # print(shift_logits.shape, shift_labels.shape, shift_labels)
            # print(f"[{torch.cuda.current_device()}] calculate loss!")
            # print(f"[{torch.cuda.current_device()}] shift_logits: {shift_logits}!")
            # print(f"[{torch.cuda.current_device()}] shift_labels: {shift_labels}")
            loss = loss_fct(shift_logits, shift_labels)
            print(f"generation loss : {loss}")
        if self.all_gather:
            input_ids, all_embs = self.all_gather_convert_ids_global2local(
                input_ids,
                target_ids,
                hard_neg_ids,
                target_attention_mask,
                hard_neg_mask,
            )
        else:
            input_ids, all_embs = self.convert_ids_global2local(
                input_ids,
                target_ids,
                hard_neg_ids,
                target_attention_mask,
                hard_neg_mask,
            )
        cnt = 0
        for item in target_idxs:
            cnt += len(item)
        if target_ids != None:
            assert cnt == target_ids.shape[0]
            question_embs = self.extract_target_embs(
                input_ids, target_idxs, outputs.last_hidden_state
            )
        else:
            assert cnt == 0
            question_embs = None
        if question_embs != None:
            scores = torch.matmul(
                question_embs, torch.transpose(all_embs, 0, 1)
            )
            print(f"Score_matrix size: {scores.size()}")

            np_logits = F.log_softmax(scores, dim=1)
            labels = torch.tensor(
                [i for i in range(np_logits.shape[0])], device=scores.device
            )
            np_loss = F.nll_loss(np_logits, labels, reduction="mean")
            print(f"np loss : {np_loss}")
            loss += np_loss / self.train_config.np_weight
            max_score, max_idxs = torch.max(np_logits, 1)
            correct_predictions_count = (
                max_idxs == torch.tensor(labels).to(max_idxs.device)
            ).sum()
            
            print(
                f"correct: {correct_predictions_count} || total:{question_embs.shape[0]} || # of ctx: {scores.shape[1]} || ret acc: {correct_predictions_count/question_embs.shape[0]}",
            )

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        **kwargs
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -1].unsqueeze(-1)

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(
                    past_state.index_select(0, beam_idx) for past_state in layer_past
                ),
            )
        return reordered_past
