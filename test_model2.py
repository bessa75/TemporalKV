from accelerate import infer_auto_device_map, init_empty_weights
import line_profiler
import copy
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Sequence, Tuple, Union, Callable
from safetensors.torch import save_file, load_file
import os

os.environ["HF_DATASETS_CACHE"] = "/data/mgiles/shil6478"
import numpy as np
import re
import math
import warnings
import torch.nn.functional as F
import torch.nn as nn
import gc

import torch
import torch.optim as optim
from torch.utils.data import Dataset


from tqdm import tqdm
import sys

# sys.path.insert(0, "/kaggle/working/KVQuant/gradients/src")
from lm_eval import evaluator
from lm_eval.models.huggingface import HFLM
from transformers import AutoModelForSequenceClassification, pipeline, AutoTokenizer
import transformers
import transformers
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache
from transformers.generation import GenerationMixin
from transformers.integrations import use_kernel_forward_from_hub
from transformers.masking_utils import create_causal_mask
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutputWithPast,
    TokenClassifierOutput,
)
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from transformers.processing_utils import Unpack
from transformers.utils import (
    TransformersKwargs,
    auto_docstring,
    can_return_tuple,
    logging,
)
from transformers.utils.generic import check_model_inputs
from transformers import Trainer
from transformers.models.llama.modeling_llama import LlamaAttention
import utils
import csv

from memory_profiler import profile

#############################################################
#############################################################
#############################################################
###### Main implementation of TemporalKV's modified    ######
###### LLM pipeline in the classes PatchLlama31_8B     ######
###### and CustomLLaMAAttention (around line 380)      ######
###### Note we also re-implement a custom DynamicCache ######
###### in transformers/cache_utils.py with a           ######
###### SparseAccumulator class to handle outliers      ######
###### Some of the functions in this file were taken   ######
###### from run-fisher2.py file, which is in part from ######
###### KVQuant's repository. When it is the case,      ######
###### we specifically mark them and do not claim      ######
###### any form of authorship over them.               ######
#############################################################
#############################################################
#############################################################


IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "</s>"
DEFAULT_UNK_TOKEN = "</s>"
PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}

PROMPT_DICT_NEW = {
    "prompt": (
        "Below is a text imitation task. You will be given a text description and asked to rewrite it in a different style.\n\n"
        "### Input:\n{input}\n\n### Output:"
    )
}


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="togethercomputer/LLaMA-2-7B-32K")
    load: Optional[str] = field(default="")
    quant_type: str = field(default="chan")
    size_centroids: int = field(default=4)
    n_bits: int = field(default=8)
    t: float = field(default=0.005)
    outlier: str = field(default="")
    all_layers: str = field(default="")
    # save_grad_path: str = field(
    #    metadata={"help": "Path to save the gradients"}
    # )


@dataclass
class DataArguments:
    dataset: str = field(default="wikitext2")
    num_examples: int = field(
        default=64, metadata={"help": "Number of calibration examples"}
    )
    seqlen: int = field(default=2048)
    maxseqlen: int = field(default=32768)
    trainset: bool = field(default=True)
    num_test_examples: int = field(default=16)
    fisher: str = field(default="no")


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    output_dir: Optional[str] = field(default=".")
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=2048,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )


def rotate_half(x):
    ### Taken from HF's Transformers library
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
    ### Taken from HF's Transformers library
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`):
            The position indices of the tokens corresponding to the query and key tensors. For example, this can be
            used to pass offsetted position ids when working with a KV-cache.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    position_ids = position_ids.cpu()
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def apply_rotary_pos_emb2(t, cos, sin, position_ids=None, unsqueeze_dim=1):
    # Custom version of HF's function for our purpose
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    t_embed = (t * cos) + (rotate_half(t) * sin)

    return t_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    ### Taken from HF's Transformers library
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    ### Taken from HF's Transformers library
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
        query.dtype
    )
    attn_weights = nn.functional.dropout(
        attn_weights, p=dropout, training=module.training
    )
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    ### Taken from HF's Transformers library
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True
        )
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True
        )

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def _tokenize_fn(
    strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer
) -> Dict:
    """Tokenize a list of strings."""
    ### Taken from KVQuant's paper in order to obtain the same data pipleine
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item()
        for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def preprocess(
    sources: Sequence[str],
    targets: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """Preprocess the data by tokenizing."""
    ### Taken from KVQuant's paper in order to obtain the same data pipleine
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [
        _tokenize_fn(strings, tokenizer) for strings in (examples, sources)
    ]
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX
    return dict(input_ids=input_ids, labels=labels)


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    ### Taken from KVQuant's paper in order to obtain the same data pipleine
    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer):
        super(SupervisedDataset, self).__init__()
        logging.warning("Loading data...")
        list_data_dict = utils.jload(data_path)

        logging.warning("Formatting inputs...")
        prompt_input, prompt_no_input = (
            PROMPT_DICT["prompt_input"],
            PROMPT_DICT["prompt_no_input"],
        )
        sources = [
            prompt_input.format_map(example)
            if example.get("input", "") != ""
            else prompt_no_input.format_map(example)
            for example in list_data_dict
        ]
        targets = [
            f"{example['output']}{tokenizer.eos_token}" for example in list_data_dict
        ]

        logging.warning("Tokenizing inputs... This may take some time...")
        data_dict = preprocess(sources, targets, tokenizer)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    ### Taken from KVQuant's paper in order to obtain the same data pipleine
    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple(
            [instance[key] for instance in instances] for key in ("input_ids", "labels")
        )
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX
        )
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )


def make_supervised_data_module(
    tokenizer: transformers.PreTrainedTokenizer, data_args
) -> Dict:
    ### Taken from KVQuant's paper in order to obtain the same data pipleine
    print("[func] make_supervised_data_module")
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = SupervisedDataset(
        tokenizer=tokenizer, data_path=data_args.data_path
    )
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(
        train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator
    )


def get_modules(layer):
    ### Taken from KVQuant's paper in order to obtain the same data pipleine
    return [
        layer.self_attn.q_proj,
        layer.self_attn.k_proj,
        layer.self_attn.v_proj,
        layer.self_attn.o_proj,
        layer.mlp.gate_proj,
        layer.mlp.up_proj,
        layer.mlp.down_proj,
    ]


def get_modules_kv(layer):
    ### Taken from KVQuant's paper in order to obtain the same data pipleine
    return [
        layer.self_attn.k_proj,
        layer.self_attn.v_proj,
    ]


def print_mem():
    mem_reserved = torch.cuda.memory_reserved("cuda") / 1024**2  # en MB
    mem_allocated = torch.cuda.memory_allocated("cuda") / 1024**2  # en MB

    print(f"GPU 0 memory reserved: {mem_reserved:.2f} MB")
    print(f"GPU 0 memory allocated: {mem_allocated:.2f} MB")

    # mem_reserved = torch.cuda.memory_reserved("cuda:1") / 1024**2  # en MB
    # mem_allocated = torch.cuda.memory_allocated("cuda:1") / 1024**2  # en MB

    # print(f"GPU 1 memory reserved: {mem_reserved:.2f} MB")
    # print(f"GPU 1 memory allocated: {mem_allocated:.2f} MB")


######################################
##### Our CUSTOM LLAMA ATTENTION #####
######################################


class CosSin:
    # This class stores the cos and sin to apply them on the fly every time the KV-Cache is loaded, avoiding to reallocate these tensors at every new token generation
    def __init__(self):
        self.global_cos = None
        self.global_sin = None


class CustomLlamaAttention(LlamaAttention):
    def __init__(
        self,
        config,
        centroids_path,
        file_name1,
        device,
        layer_idx,
        file_name2,
        source_attn,
        n_layers,
        cos_sin,
        outlier,
        pretraining_tp=1,
        quant_type="norm",
        quant_type2="",
        t=0.005,
    ):
        super().__init__(config, layer_idx)
        self.outlier = outlier
        self.to(torch.bfloat16)
        self.quant_type = quant_type
        self.quant_type2 = quant_type2
        self.cos_sin = cos_sin
        self.n_layers = n_layers
        self.key_mses = torch.zeros((self.num_key_value_heads, self.head_dim)).to(
            device
        )
        self.value_mses = torch.zeros((self.num_key_value_heads, self.head_dim)).to(
            device
        )

        self.n_samples = 0
        self.t = t
        # Loading centroids
        if quant_type in ["norm", "kvquant", "kvquant1p", "chan"]:
            centroids = load_file(centroids_path + quant_type + "/" + file_name1)
            self.centroids_v = (
                torch.Tensor(centroids["v"]).to(device).to(torch.bfloat16)
            )  # (num_heads,self.head_dim//size_centroids,n_centroids,size_centroids) ou (num_heads,self.head_dim//size_centroids,n_centroids,size_centroids)
            self.centroids_k = (
                torch.Tensor(centroids["k"]).to(device).to(torch.bfloat16)
            )
            assert self.centroids_v.shape == self.centroids_k.shape
            assert self.num_key_value_heads == self.centroids_v.shape[0]
            # print(self.centroids_v.shape,self.head_dim)

            self.size_centroids = self.centroids_v.shape[-1]
            self.n_centroids = self.centroids_v.shape[-2]

            if quant_type in ["kvquant", "kvquant1p"]:
                assert self.size_centroids == 1
                interleave = torch.repeat_interleave(
                    torch.arange(self.centroids_v.shape[1]),
                    self.head_dim // self.centroids_v.shape[1],
                )
                self.centroids_v = self.centroids_v[:, interleave].contiguous()
                self.centroids_k = self.centroids_k[:, interleave].contiguous()
            else:
                assert self.size_centroids == int(
                    re.search(r"size(\d+)bit", file_name1).group(1)
                )

            assert (
                self.head_dim == self.centroids_v.shape[1] * self.centroids_v.shape[3]
            )
            assert self.layer_idx == int(
                re.search(r"layer(\d+)size", file_name1).group(1)
            )
        else:
            self.size_centroids = -1
        self.dic_v = {}
        self.dic_k = {}
        self.pretraining_tp = pretraining_tp
        self.device = device

        # Preparing normalization statistics / quantiles for outlier removal
        if quant_type != "noquant":
            dic_stats = load_file(centroids_path + "stats/" + file_name2)
            self.key_means = dic_stats["mean_k"].to(device).to(torch.bfloat16)
            self.key_stds = dic_stats["stds_k"].to(device).to(torch.bfloat16)
            self.value_means = dic_stats["mean_v"].to(device).to(torch.bfloat16)
            self.value_stds = dic_stats["stds_v"].to(device).to(torch.bfloat16)
            if self.outlier:
                self.key_quantiles = (
                    dic_stats["key_quantiles"].to(device).to(torch.bfloat16)
                )
            assert len(self.key_means.shape) == 4
            assert self.num_key_value_heads == self.key_means.shape[1]
            assert self.head_dim == self.key_means.shape[3]
        else:
            self.key_means = None
            self.key_stds = None
            self.value_means = None
            self.value_stds = None

        self.load_state_dict(source_attn.state_dict(), strict=True)
        # for (n1, p1), (n2, p2) in zip(self.named_parameters(), source_attn.named_parameters()):
        #     if not torch.allclose(p1, p2, atol=1e-6):
        #         print(f"Mismatch: {n1} != {n2}")
        self.training = source_attn.training
        self.eval() if not source_attn.training else self.train()
        self.generated_tokens = 0
        self.offset = None
        self.to(device)
        self.value_scale = None
        # for attr_name, attr_value in vars(source_attn).items():
        # setattr(self, attr_name, attr_value)

    @line_profiler.profile
    def encode(self, states, var_type, stride=1):

        """
        Encoding function of keys and values
        Input : torch.Tensor with dtype torch.float16 of shape (bsz, self.num_key_value_heads, q_len,  self.head_dim)
        Output : torch.Tensor with dtype torch.int8/16 (bsz, self.num_key_value_heads,q_len,  self.head_dim//size_centroids) for chan
        or torch.Tensor with dtype torch.int8/16 (bsz, self.num_key_value_heads, q_len//size_centroids,  self.head_dim//size_centroids, size_centroids) for norm and kvquant
        """

        ratio = states.shape
        if self.quant_type == "chan":
            if states.shape[0] * states.shape[2] >= 20000:
                codes = torch.empty(
                    (*states.shape[0:3], self.head_dim // self.size_centroids),
                    device=self.device,
                    dtype=torch.int8,
                )  # int16
                for i in range(0, self.num_key_value_heads, stride):
                    statesi = states[:, i : i + stride].view(
                        states.shape[0],
                        stride,
                        states.shape[2],
                        self.head_dim // self.size_centroids,
                        self.size_centroids,
                        1,
                    )  # (bsz,stride, q_len, self.head_dim//size_centroids, size_centroids,1)
                    if var_type == "v":
                        centroids = (
                            self.centroids_v[i : i + stride].unsqueeze(0).unsqueeze(2)
                        )  # (1,stride,1,self.head_dim//size_centroids,n_centroids,size_centroids)
                    elif var_type == "k":
                        centroids = (
                            self.centroids_k[i : i + stride].unsqueeze(0).unsqueeze(2)
                        )  # (1,stride,1,self.head_dim//size_centroids,n_centroids,size_centroids)
                    else:
                        raise Exception("wrong var_type, neither v or k")
                    dps = torch.matmul(centroids, statesi).squeeze(
                        -1
                    )  # (bsz, stride, q_len, self.head_dim//size_centroids, n_centroids)
                    distances = (
                        (torch.norm(centroids, dim=-1) ** 2)
                        - 2 * dps
                        + (torch.norm(statesi, dim=-2) ** 2)
                    )
                    codesi = (distances.argmin(dim=-1) - 128).to(
                        torch.int8
                    )  # int16 #(bsz, stride, q_len, self.head_dim//size_centroids)
                    codes[:, i : i + stride] = codesi
            else:
                states = states.view(
                    states.shape[0],
                    states.shape[1],
                    states.shape[2],
                    self.head_dim // self.size_centroids,
                    self.size_centroids,
                    1,
                )  # (bsz, num_heads,q_len, self.head_dim//size_centroids, size_centroids,1)
                if var_type == "v":
                    centroids = self.centroids_v.unsqueeze(0).unsqueeze(
                        2
                    )  # (1,num_heads,1,self.head_dim//size_centroids,n_centroids,size_centroids)
                elif var_type == "k":
                    centroids = self.centroids_k.unsqueeze(0).unsqueeze(
                        2
                    )  # (1,num_heads,1,self.head_dim//size_centroids,n_centroids,size_centroids)
                else:
                    raise Exception("wrong var_type, neither v or k")
                # dps = torch.matmul(centroids,states).squeeze(-1) #(bsz, num_heads,q_len, self.head_dim//size_centroids, n_centroids)
                codes = (
                    (
                        (torch.norm(centroids, dim=-1) ** 2)
                        - 2 * torch.matmul(centroids, states).squeeze(-1)
                        + torch.norm(states, dim=-2) ** 2
                    ).argmin(dim=-1)
                    - 128
                ).to(
                    torch.int8
                )  # -128#int8 #(bsz, num_heads, q_len, self.head_dim//size_centroids)
        elif (
            self.quant_type == "norm"
            or self.quant_type == "kvquant"
            or self.quant_type == "kvquant1p"
        ):
            # print(f"calculated qlen : {states.shape[2]*self.size_centroids}")
            # print(f"size_centroids : {self.size_centroids}")
            # print(f"q_len : {states.shape[2]}")

            if states.shape[0] * states.shape[2] >= 20000:
                codes = torch.empty(
                    (
                        *states.shape[0:2],
                        states.shape[2] // self.size_centroids,
                        self.head_dim // self.size_centroids,
                        self.size_centroids,
                    ),
                    device=self.device,
                    dtype=torch.int8,
                )  # int8
                for i in range(0, self.num_key_value_heads, stride):
                    statesi = (
                        states[:, i : i + stride]
                        .view(
                            states.shape[0],
                            stride,
                            states.shape[2] // self.size_centroids,
                            self.size_centroids,
                            self.head_dim,
                            1,
                        )
                        .transpose(3, 4)
                    )  # (bsz, stride, q_len//size_centroids, self.head_dim, size_centroids,1)
                    statesi = statesi.reshape(
                        statesi.shape[0],
                        statesi.shape[1],
                        statesi.shape[2],
                        statesi.shape[3] // self.size_centroids,
                        self.size_centroids,
                        self.size_centroids,
                        1,
                    )  # (bsz,stride,q_len//size_centroids,self.head_dim//size_centroids,size_centroids,size_centroids,1)
                    if var_type == "v":
                        centroids = (
                            self.centroids_v[i : i + stride]
                            .unsqueeze(1)
                            .unsqueeze(0)
                            .unsqueeze(4)
                        )  # (1,stride,1,self.head_dim//size_centroids,1,n_centroids,size_centroids)
                    elif var_type == "k":
                        centroids = (
                            self.centroids_k[i : i + stride]
                            .unsqueeze(1)
                            .unsqueeze(0)
                            .unsqueeze(4)
                        )  # (1,stride,1,self.head_dim//size_centroids,1,n_centroids,size_centroids)
                    # print(self.layer_idx,i)
                    # print_mem()
                    codesi = (
                        (
                            torch.norm(centroids, dim=-1) ** 2
                            - 2 * torch.matmul(centroids, statesi).squeeze(-1)
                            + torch.norm(statesi, dim=-2) ** 2
                        ).argmin(dim=-1)
                        - 128
                    ).to(
                        torch.int8
                    )  # -128#int8 #(bsz,stride,q_len//size_centroids,self.head_dim//size_centroids,size_centroids)
                    codes[:, i : i + stride] = codesi
            else:
                states = states.view(
                    states.shape[0],
                    states.shape[1],
                    states.shape[2] // self.size_centroids,
                    self.size_centroids,
                    self.head_dim,
                    1,
                ).transpose(
                    3, 4
                )  # (bsz, n_key_value_heads, q_len//size_centroids, self.head_dim, size_centroids,1)
                states = states.reshape(
                    states.shape[0],
                    states.shape[1],
                    states.shape[2],
                    states.shape[3] // self.size_centroids,
                    self.size_centroids,
                    self.size_centroids,
                    1,
                )  # (bsz, n_key_value_heads, q_len//size_centroids,self.head_dim//size_centroids,size_centroids,size_centroids,1)
                if var_type == "v":
                    centroids = (
                        self.centroids_v.unsqueeze(1).unsqueeze(0).unsqueeze(4)
                    )  # (1,num_key_value_heads,1,self.head_dim//size_centroids,1,n_centroids,size_centroids)
                elif var_type == "k":
                    centroids = (
                        self.centroids_k.unsqueeze(1).unsqueeze(0).unsqueeze(4)
                    )  # (1,1,self.head_dim//size_centroids,1,n_centroids,size_centroids)
                # dps=torch.matmul(centroids,states).squeeze(-1) #(bsz,num_key_value_heads,q_len//size_centroids,self.head_dim//size_centroids,size_centroids,n_centroids)
                codes = (
                    (
                        torch.norm(centroids, dim=-1) ** 2
                        - 2 * torch.matmul(centroids, states).squeeze(-1)
                        + torch.norm(states, dim=-2) ** 2
                    ).argmin(dim=-1)
                    - 128
                ).to(
                    torch.int8
                )  # -128 #int8 #(bsz,num_key_value_heads,1,q_len//size_centroids,self.head_dim//size_centroids,size_centroids)

        return codes

    @line_profiler.profile
    def decode(self, codes, var_type, seen_tokens, offset=None):

        """
        Decoding function of keys and values centroid indices
        Input : torch.Tensor with dtype torch.int8/16 (bsz, self.num_key_value_heads,q_len,  self.head_dim//size_centroids) for chan
        or torch.Tensor with dtype torch.int8/16 (bsz, self.num_key_value_heads, q_len//size_centroids,  self.head_dim//size_centroids, size_centroids) for norm and kvquant
        Output : torch.Tensor with dtype torch.float16 of shape (bsz, self.num_key_value_heads, q_len,  self.head_dim)

        """

        decoded_states = None
        codes = codes.to(torch.int16) + 128
        if self.quant_type == "chan":
            """
            ### Initial tiled solution with advanced indexing, which was not efficient enough
            for i in range (0,self.num_key_value_heads):
                codesi=codes[:,i] #(bsz,q_len,self.head_dim//size_centroids)
                if var_type=="v":
                    centroids=self.centroids_v[i] #(self.head_dim//size_centroids,n_centroids,size_centroids)
                elif var_type=="k":
                    centroids=self.centroids_k[i] #(self.head_dim//size_centroids,n_centroids,size_centroids)
                else:
                    raise Exception("wrong var_type, neither v or k")
                indices=torch.arange(0,codesi.shape[2]).unsqueeze(0).unsqueeze(0)
                #centroids[i,codes[j,k,i]] for all j,k,i
                decoded_statesi=centroids[indices,codesi].view(codesi.shape[0],codesi.shape[1],self.head_dim).unsqueeze(1) #(bsz,1,q_len,self.head_dim//size_centroids,size_centroids)
                if decoded_states is None:
                    decoded_states = decoded_statesi
                else:
                    decoded_states = torch.cat((decoded_states,decoded_statesi),dim=1)
            return decoded_states
            """
            codes = codes  # (bsz,num_heads,q_len,self.head_dim//size_centroids)
            if var_type == "v":
                centroids = (
                    self.centroids_v
                )  # (num_heads,self.head_dim//size_centroids,n_centroids,size_centroids)
            elif var_type == "k":
                centroids = (
                    self.centroids_k
                )  # (num_heads,self.head_dim//size_centroids,n_centroids,size_centroids)
            else:
                raise Exception("wrong var_type, neither v or k")

            bsz, _, len, _ = codes.shape
            codes = (
                codes.transpose(1, 2)
                .reshape(
                    bsz * len,
                    self.num_key_value_heads,
                    self.head_dim // self.size_centroids,
                )
                .unsqueeze(-1)
            )
            codes = codes.expand(
                bsz * len,
                self.num_key_value_heads,
                self.head_dim // self.size_centroids,
                self.size_centroids,
            )
            codes = codes.to(torch.long)
            centroids = centroids.permute(
                2, 0, 1, 3
            )  # (n_centroids,num_heads,self.head_dim//size_centroids,size_centroids)
            if seen_tokens > 0:
                decoded_states_tot = torch.empty(
                    (bsz, self.num_key_value_heads, len + seen_tokens, self.head_dim),
                    dtype=torch.bfloat16,
                    device=self.device,
                )
                decoded_states_tot[:, :, :len] = (
                    torch.gather(centroids, 0, codes)
                    .view(bsz, len, self.num_key_value_heads, self.head_dim)
                    .permute(0, 2, 1, 3)
                )
                # decoded_states=centroids[indices_num_head,indices_head_dim,codes.to(torch.int)].view(codes.shape[0],codes.shape[1],codes.shape[2],self.head_dim) #(bsz,num_heads,q_len,self.head_dim//size_centroids,size_centroids)
                return decoded_states_tot
            else:
                decoded_states = (
                    torch.gather(centroids, 0, codes)
                    .view(bsz, len, self.num_key_value_heads, self.head_dim)
                    .permute(0, 2, 1, 3)
                )
                return decoded_states
        elif self.quant_type in ["norm", "kvquant", "kvquant1p"]:
            """
            ### Initial tiled solution with advanced indexing, which was not efficient enough
            for i in range (0,self.num_key_value_heads):
                codesi=codes[:,i] #(bsz,q_len//size_centroids,self.head_dim//size_centroids,size_centroids)
                if var_type=="v":
                    centroids=self.centroids_v[i] #(self.head_dim//size_centroids,n_centroids,size_centroids)
                elif var_type=="k":
                    centroids=self.centroids_k[i] #(self.head_dim//size_centroids,n_centroids,size_centroids)
                else:
                    raise Exception("wrong var_type, neither v or k")
                indices=torch.arange(0,codesi.shape[2]).unsqueeze(0).unsqueeze(0).unsqueeze(-1)
                #centroids[i,codes[j,k,i]] for all j,k,i
                decoded_statesi=centroids[indices,codesi].view(codesi.shape[0],codesi.shape[1],self.head_dim,self.size_centroids)#(bsz,q_len//size_centroids,self.head_dim//size_centroids,size_centroids,size_centroids)
                decoded_statesi=decoded_statesi.transpose(2,3).reshape(codesi.shape[0],codesi.shape[1]*codesi.shape[3],self.head_dim).unsqueeze(1) #(bsz,1,q_len,self.head_dim)
                if decoded_states is None:
                    decoded_states = decoded_statesi
                else:
                    decoded_states = torch.cat((decoded_states,decoded_statesi),dim=1)
            return decoded_states
            """
            codes = codes  # (bsz,num_heads,q_len//size_centroids,self.head_dim//size_centroids,size_centroids)
            if var_type == "v":
                centroids = (
                    self.centroids_v
                )  # (num_heads,self.head_dim//size_centroids,n_centroids,size_centroids)
            elif var_type == "k":
                centroids = (
                    self.centroids_k
                )  # (num_heads,self.head_dim//size_centroids,n_centroids,size_centroids)
            else:
                raise Exception("wrong var_type, neither v or k")

            indices_head_dim = (
                torch.arange(0, codes.shape[3])
                .unsqueeze(0)
                .unsqueeze(0)
                .unsqueeze(0)
                .unsqueeze(-1)
            )  # (1,1,1,self.head_dim//size_centroids,1)
            indices_num_head = (
                torch.arange(0, codes.shape[1])
                .unsqueeze(0)
                .unsqueeze(2)
                .unsqueeze(2)
                .unsqueeze(-1)
            )  # (1,num_heads,1,1,1)

            bsz, _, len, _, _ = codes.shape
            codes = (
                codes.transpose(1, 2)
                .reshape(
                    bsz * len,
                    self.num_key_value_heads,
                    self.head_dim // self.size_centroids,
                    self.size_centroids,
                )
                .unsqueeze(-1)
            )
            codes = codes.expand(
                bsz * len,
                self.num_key_value_heads,
                self.head_dim // self.size_centroids,
                self.size_centroids,
                self.size_centroids,
            )
            centroids = centroids.permute(2, 0, 1, 3).unsqueeze(
                -2
            )  # (n_centroids,num_heads,self.head_dim//size_centroids,1,size_centroids)
            centroids = centroids.expand(
                self.n_centroids,
                self.num_key_value_heads,
                self.head_dim // self.size_centroids,
                self.size_centroids,
                self.size_centroids,
            )

            totallen = len * self.size_centroids + seen_tokens
            offset2 = 0
            if self.quant_type == "norm":
                assert offset is not None
                offset2 = offset
            totallen += offset2

            codes = codes.to(torch.long)

            
            ### Initial tiled solution with advanced indexing, which was not efficient enough
            # decoded_states=centroids[indices_num_head,indices_head_dim,codes].view(codes.shape[0],codes.shape[1],codes.shape[2],self.head_dim,self.size_centroids) #(bsz,num_heads,q_len//size_centroids,self.head_dim,size_centroids)
            decoded_states = (
                torch.gather(centroids, 0, codes)
                .view(
                    bsz,
                    len,
                    self.num_key_value_heads,
                    self.head_dim,
                    self.size_centroids,
                )
                .permute(0, 2, 1, 4, 3)
            )
            decoded_states = decoded_states.contiguous().flatten(2, 3)

            decoded_states_tot = torch.empty(
                (bsz, self.num_key_value_heads, totallen, self.head_dim),
                dtype=torch.bfloat16,
                device=self.device,
            )

            decoded_states_tot[
                :, :, offset2 : offset2 + len * self.size_centroids
            ] = decoded_states
            # decoded_states=decoded_states.transpose(3,4).reshape(codes.shape[0],codes.shape[1],codes.shape[2]*self.size_centroids,self.head_dim)
            return decoded_states_tot

    def normalize(self, states, var_type, use_dicstats=True):

        """
        # Input : torch.Tensor with dtype torch.float16 (bsz, self.num_key_value_heads, q_len,  self.head_dim)
        # Output : normalized torch.Tensor with dtype torch.float16 of same dimension as the input
        # key/value_means/scale has shape (1,self.num_key_value_heads,1,head_dim)
        """

        if self.quant_type == "chan":
            return states

        if var_type == "k":
            if self.quant_type != "kvquant1p":
                if not use_dicstats:
                    self.key_means = torch.mean(states, dim=(0, 2), keepdim=True)
                    self.key_stds = torch.std(
                        states - self.key_means, dim=(0, 2), keepdim=True
                    )
                norm_states = (states - self.key_means) / self.key_stds
            else:
                lower_quantile = self.key_quantiles[0].reshape(
                    1, self.num_key_value_heads, 1, self.head_dim
                )
                upper_quantile = self.key_quantiles[1].reshape(
                    1, self.num_key_value_heads, 1, self.head_dim
                )
                self.key_scale = (upper_quantile - lower_quantile) / 2
                norm_states = (states - self.key_means) / self.key_scale

        elif var_type == "v":
            if self.quant_type == "kvquant":
                self.value_means = states.mean(dim=(1, 3), keepdim=True)
                self.value_stds = states.std(dim=(1, 3), keepdim=True)
                norm_states = (states - self.value_means) / self.value_stds
            elif self.quant_type != "kvquant1p":
                norm_states = (states - self.value_means) / self.value_stds
            else:
                value_states_reshaped = states.transpose(1, 2).view(
                    states.shape[0] * states.shape[2], states.shape[1] * states.shape[3]
                )
                # print(value_states_reshaped.dtype)
                value_lower_quantiles = (
                    torch.quantile(value_states_reshaped.float(), self.t, dim=1)
                    .reshape(states.shape[0], 1, states.shape[2], 1)
                    .to(torch.bfloat16)
                )
                value_upper_quantiles = (
                    torch.quantile(value_states_reshaped.float(), 1 - self.t, dim=1)
                    .reshape(states.shape[0], 1, states.shape[2], 1)
                    .to(torch.bfloat16)
                )
                value_scale = (value_upper_quantiles - value_lower_quantiles) / 2
                if states.shape[2] > 1:
                    self.value_scale = value_scale
                else:
                    self.value_scale = torch.cat([self.value_scale, value_scale], dim=2)
                norm_states = (states - self.value_means) / self.value_scale
        else:
            raise Exception("wrong var_type, neither v or k")

        return norm_states

    def denormalize(self, norm_states, var_type):

        """
        # Input : normalized decompressed torch.Tensor with dtype torch.float16 (bsz, self.num_key_value_heads, q_len,  self.head_dim)
        # Output : denormalized decompressed torch.Tensor with dtype torch.float16 of same dimension as the input
        # key/value_means/scale has shape (1,self.num_key_value_heads,1,head_dim)
        """

        if self.quant_type == "chan":
            return norm_states
        if self.quant_type == "kvquant1p":
            if var_type == "k":
                denorm_states = norm_states * self.key_scale + self.key_means
            elif var_type == "v":
                denorm_states = norm_states * self.value_scale + self.value_means
        else:
            if var_type == "k":
                denorm_states = norm_states * self.key_stds + self.key_means
            elif var_type == "v":
                denorm_states = norm_states * self.value_stds + self.value_means
            else:
                raise Exception("wrong var_type, neither v or k")
        return denorm_states


class PatchLlama31_8B(CustomLlamaAttention):
    @line_profiler.profile
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        
        input_shape = hidden_states.shape[:-1]
        bsz, q_len, _ = hidden_states.size()

        ### Already existing Llama code
        if (
            self.config.pretraining_tp > 1
        ):  # does this parameter exist ? else initialize
            key_value_slicing = (
                self.num_key_value_heads * self.head_dim
            ) // self.config.pretraining_tp
            query_slices = self.q_proj.weight.split(
                (self.num_key_value_heads * self.head_dim)
                // self.config.pretraining_tp,
                dim=0,
            )
            key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
            value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

            query_states = [
                F.linear(hidden_states, query_slices[i])
                for i in range(self.config.pretraining_tp)
            ]
            query_states = torch.cat(query_states, dim=-1)

            key_states = [
                F.linear(hidden_states, key_slices[i])
                for i in range(self.config.pretraining_tp)
            ]
            key_states = torch.cat(key_states, dim=-1)

            value_states = [
                F.linear(hidden_states, value_slices[i])
                for i in range(self.config.pretraining_tp)
            ]
            value_states = torch.cat(value_states, dim=-1)

        else:
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

        query_states = query_states.view(
            bsz, q_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        key_states = key_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)
        value_states = value_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)

        # Handling of global cos sin to avoid recomputing them every time
        cos, sin = position_embeddings
        if self.layer_idx == 0:
            if cos.shape[1] > 1:
                self.cos_sin.global_cos, self.cos_sin.global_sin = cos, sin
            else:
                assert self.cos_sin.global_cos.shape[1] > 1
                self.cos_sin.global_cos = torch.cat(
                    [self.cos_sin.global_cos, cos], dim=1
                )
                self.cos_sin.global_sin = torch.cat(
                    [self.cos_sin.global_sin, sin], dim=1
                )

        #######################################################
        #######################################################
        ####  Loading from KV-Cache : heart of our method  ####
        #######################################################
        #######################################################

        if (
            q_len == 1 and self.layer_idx == 0
        ):  # we dont compress generated tokens to keep fair comparison with kvquant and channelkv
            assert past_key_value is not None
            past_key_value.nb_local_tokens = 130
            past_key_value.seen_tokens += 1

        if q_len > 1:
            self.offset = q_len % self.size_centroids
        if self.quant_type in ["chan", "norm", "kvquant", "kvquant1p"]:
            assert past_key_value is not None
            if past_key_value is not None:
                if past_key_value.n_layers is None:
                    past_key_value.n_layers = self.n_layers
                    past_key_value.init_n_layers(
                        bsz,
                        q_len,
                        self.num_key_value_heads,
                        self.head_dim,
                        self.outlier,
                        self.t,
                        self.quant_type,
                        self.size_centroids,
                        self.offset,
                    )
            if self.outlier:
                if past_key_value.key_lower_quantiles[self.layer_idx] is None:
                    past_key_value.key_lower_quantiles[
                        self.layer_idx
                    ] = self.key_quantiles[0].reshape(
                        1, self.num_key_value_heads, 1, self.head_dim
                    )
                    past_key_value.key_upper_quantiles[
                        self.layer_idx
                    ] = self.key_quantiles[1].reshape(
                        1, self.num_key_value_heads, 1, self.head_dim
                    )
            assert self.offset is not None

            # normalize keys and values
            norm_key_states = self.normalize(key_states, var_type="k")
            norm_value_states = self.normalize(value_states, var_type="v")

            if self.quant_type == "norm":
                # if decode we dont do anything here
                # at prefilling we encode the context dividable by size_centroids
                if q_len == 1:
                    key_codes, value_codes = None, None
                else:
                    key_codes = self.encode(
                        norm_key_states[:, :, self.offset :], var_type="k"
                    )
                    value_codes = self.encode(
                        norm_value_states[:, :, self.offset :], var_type="v"
                    )
            else:
                if q_len == 1:
                    key_codes, value_codes = None, None
                else:
                    key_codes = self.encode(norm_key_states, var_type="k")
                    value_codes = self.encode(norm_value_states, var_type="v")

            if past_key_value is not None:
                cache_kwargs = {
                    "sin": sin,
                    "cos": cos,
                    "key_codes": key_codes,
                    "value_codes": value_codes,
                    "n_centroids": self.n_centroids,
                }  # Specific to RoPE models
                if self.outlier:
                    (
                        key_codes,
                        value_codes,
                        key_sparse,
                        value_sparse,
                        sink_tokens_k,
                        sink_tokens_v,
                        local_tokens_k,
                        local_tokens_v,
                    ) = past_key_value.update(
                        key_states, value_states, self.layer_idx, cache_kwargs
                    )
                else:
                    (
                        key_codes,
                        value_codes,
                        sink_tokens_k,
                        sink_tokens_v,
                        local_tokens_k,
                        local_tokens_v,
                    ) = past_key_value.update(
                        key_states, value_states, self.layer_idx, cache_kwargs
                    )
            # print("here1")
            if self.quant_type == "norm":
                norm_key_states = self.decode(
                    key_codes,
                    var_type="k",
                    seen_tokens=past_key_value.seen_tokens,
                    offset=self.offset,
                )
                norm_value_states = self.decode(
                    value_codes,
                    var_type="v",
                    seen_tokens=past_key_value.seen_tokens,
                    offset=self.offset,
                )
            else:
                norm_key_states = self.decode(
                    key_codes, var_type="k", seen_tokens=past_key_value.seen_tokens
                )
                norm_value_states = self.decode(
                    value_codes, var_type="v", seen_tokens=past_key_value.seen_tokens
                )

            # print("here2")
            # denormalize keys and values if norm
            denorm_key_states = self.denormalize(norm_key_states, var_type="k")
            denorm_value_states = self.denormalize(norm_value_states, var_type="v")
            # print("here3")

            # Outlier removal
            if self.outlier:
                idx_key = tuple(key_sparse[0][i] for i in range(key_sparse[0].shape[0]))
                idx_value = tuple(
                    value_sparse[0][i] for i in range(value_sparse[0].shape[0])
                )

                denorm_key_states.index_put_(idx_key, key_sparse[1], accumulate=False)
                denorm_value_states.index_put_(
                    idx_value, value_sparse[1], accumulate=False
                )

            # reintegrate local and sink tokens
            if past_key_value is not None:
                denorm_key_states[:, :, : sink_tokens_k.shape[2]] = sink_tokens_k
                denorm_value_states[:, :, : sink_tokens_v.shape[2]] = sink_tokens_v
                if local_tokens_k is not None:
                    assert local_tokens_v is not None
                    # print(local_tokens_k.shape,past_key_value.seen_tokens,denorm_key_states.shape,key_states.shape)
                    assert local_tokens_k.shape[2] == past_key_value.seen_tokens
                    assert denorm_key_states.shape == denorm_value_states.shape
                    # if self.layer_idx==2:
                    #     print(f"appending {past_key_value.seen_tokens} local tokens")
                    denorm_key_states[
                        :, :, denorm_key_states.shape[2] - local_tokens_k.shape[2] :
                    ] = local_tokens_k.contiguous()
                    denorm_value_states[
                        :, :, denorm_value_states.shape[2] - local_tokens_v.shape[2] :
                    ] = local_tokens_v.contiguous()

            else:
                raise Exception("no KV cache")

            query_states = apply_rotary_pos_emb2(query_states, cos, sin)
            key_states = apply_rotary_pos_emb2(
                denorm_key_states, self.cos_sin.global_cos, self.cos_sin.global_sin
            )
            value_states = denorm_value_states

        elif self.quant_type == "noquant":
            # if self.layer_idx==0 and past_key_value is not None:
            #     print(f"Generated_tokens : {past_key_value.seen_tokens}")
            query_states = apply_rotary_pos_emb2(query_states, cos, sin)
            key_states = apply_rotary_pos_emb2(key_states, cos, sin)

            if past_key_value is not None:
                if past_key_value.n_layers is None:
                    past_key_value.n_layers = self.n_layers
                    past_key_value.init_n_layers(
                        bsz,
                        q_len,
                        self.num_key_value_heads,
                        self.head_dim,
                        self.outlier,
                        self.t,
                        self.quant_type,
                        self.size_centroids,
                        self.offset,
                    )
                cache_kwargs = {
                    "sin": sin,
                    "cos": cos,
                    "cache_position": cache_position,
                }
                key_states, value_states = past_key_value.update(
                    key_states, value_states, self.layer_idx, cache_kwargs
                )
        else:
            raise ValueError("quant type unidentified")

        # print(f"attention mask : {attention_mask}")
        # print(query_states.shape,key_states.shape,value_states.shape,attention_mask.shape)

        ########################################################################################################
        ########################################################################################################
        ##### Finishing the forward pass (following code is not modified from the original implementation) #####
        ########################################################################################################
        ########################################################################################################

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[
                self.config._attn_implementation
            ]

        # print_mem()
        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


## Patch for older Transformers version (Python 3.9, SDPAattention)
class PatchLlamaOld(CustomLlamaAttention):
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:

        bsz, q_len, _ = hidden_states.size()

        if self.config.pretraining_tp > 1:
            key_value_slicing = (
                self.num_key_value_heads * self.head_dim
            ) // self.config.pretraining_tp
            query_slices = self.q_proj.weight.split(
                (self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0
            )
            key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
            value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

            query_states = [
                F.linear(hidden_states, query_slices[i])
                for i in range(self.config.pretraining_tp)
            ]
            query_states = torch.cat(query_states, dim=-1)

            key_states = [
                F.linear(hidden_states, key_slices[i])
                for i in range(self.config.pretraining_tp)
            ]
            key_states = torch.cat(key_states, dim=-1)

            value_states = [
                F.linear(hidden_states, value_slices[i])
                for i in range(self.config.pretraining_tp)
            ]
            value_states = torch.cat(value_states, dim=-1)

        else:
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

        query_states = query_states.view(
            bsz, q_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        key_states = key_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)
        value_states = value_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)

        #########################
        ### Start of our code ###
        #########################

        if self.quant_type in ["chan", "norm", "kvquant"]:
            # normalize keys and values

            norm_key_states = self.normalize(key_states, var_type="k")
            norm_value_states = self.normalize(value_states, var_type="v")

            # put local and sink tokens apart
            key_codes = self.encode(norm_key_states, var_type="k")
            value_codes = self.encode(norm_value_states, var_type="v")

            kv_seq_len = key_states.shape[-2]
            if past_key_value is not None:
                if past_key_value.n_layers is None:
                    past_key_value.n_layers = self.n_layers
                    past_key_value.init_n_layers()
                if self.layer_idx is None:
                    raise ValueError(
                        f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                        "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                        "with a layer index."
                    )
                kv_seq_len += past_key_value.get_usable_length(
                    kv_seq_len, self.layer_idx
                )

            cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
            if past_key_value is not None:
                cache_kwargs = {
                    "sin": sin,
                    "cos": cos,
                    "key_codes": key_codes,
                    "value_codes": value_codes,
                }  # Specific to RoPE models
                (
                    key_codes,
                    value_codes,
                    sink_tokens_k,
                    sink_tokens_v,
                    local_tokens_k,
                    local_tokens_v,
                ) = past_key_value.update(
                    key_states, value_states, self.layer_idx, cache_kwargs
                )

            norm_key_states = self.decode(key_codes, var_type="k")
            norm_value_states = self.decode(value_codes, var_type="v")

            # denormalize keys and values if norm
            denorm_key_states = self.denormalize(norm_key_states, var_type="k")
            denorm_value_states = self.denormalize(norm_value_states, var_type="v")

            # reintegrate local and sink tokens
            if past_key_value is not None:
                denorm_key_states[:, :, : sink_tokens_k.shape[2]] = sink_tokens_k
                denorm_value_states[:, :, : sink_tokens_v.shape[2]] = sink_tokens_v
                if local_tokens_k is not None:
                    assert local_tokens_v is not None
                    denorm_key_states[:, :, -local_tokens_k.shape[2] :] = local_tokens_k
                    denorm_value_states[
                        :, :, -local_tokens_v.shape[2] :
                    ] = local_tokens_v
            else:
                raise Exception("no KV cache")

            query_states, key_states = apply_rotary_pos_emb(
                query_states, denorm_key_states, cos, sin, position_ids
            )
            value_states = denorm_value_states
        elif self.quant_type == "noquant":
            kv_seq_len = key_states.shape[-2]
            if past_key_value is not None:
                kv_seq_len += past_key_value.get_usable_length(
                    kv_seq_len, self.layer_idx
                )
                if past_key_value.n_layers is None:
                    past_key_value.n_layers = self.n_layers
                    past_key_value.init_n_layers()
            cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)

            query_states, key_states = apply_rotary_pos_emb(
                query_states, key_states, cos, sin, position_ids
            )

            if past_key_value is not None:
                cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
                key_states, value_states = past_key_value.update(
                    key_states, value_states, self.layer_idx, cache_kwargs
                )
        else:
            raise ValueError("quant type unidentified")

        #######################
        ### End of our code ###
        #######################

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )

        # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
        # Reference: https://github.com/pytorch/pytorch/issues/112577.
        if query_states.device.type == "cuda" and attention_mask is not None:
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=attention_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            # The q_len > 1 is necessary to match with AttentionMaskConverter.to_causal_4d that does not create a causal mask in case q_len == 1.
            is_causal=self.is_causal and attention_mask is None and q_len > 1,
        )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        return attn_output, None, past_key_value


#####################
#####################
##### Main loop #####
#####################
#####################


def train():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    (
        model_args,
        data_args,
        training_args,
    ) = parser.parse_args_into_dataclasses()  # remove these args
    if data_args.dataset in ["c4", "ptb", "wikitext2", "ptbnew", "c4new"]:
        from datautils import get_loaders

        print("Calibration with " + data_args.dataset)
        dataloader, testloader = get_loaders(
            data_args.dataset,
            model=model_args.model_name_or_path,
            seqlen=data_args.seqlen,
            seed=0,
        )
        is_perplexity = True
    elif data_args.dataset in [
        "ruler_fwe",
        "ruler_vt",
        "niah_multiquery",
        "niah_multivalue",
        "niah_multikey_1",
        "niah_multikey_2",
        "niah_multikey_3",
        "ruler_cwe",
        "winogrande",
        "arc_challenge",
        "piqa",
        "hellaswag",
        "swag",
        "rte",
        "mmlu",
        "openbookqa",
        "niah_single_1",
        "niah_single_2",
        "niah_single_3",
        "csqa",
        "strategyqa",
        "truthfulqa_mc",
        "logiqa",
        "pubmed_qa",
        "lambada_standard",
        "lambada_openai",
        "headqa_es",
        "mathqa",
        "ruler_qa_hotpot",
        "ruler_qa_squad",
    ]:
        is_perplexity = False
    else:
        raise NotImplementedError("Please define your own dataset here")

    import math
    from transformers import AutoConfig

    config = transformers.AutoConfig.from_pretrained(model_args.model_name_or_path)

    if model_args.model_name_or_path == "togethercomputer/LLaMA-2-7B-32K":
        context_size = data_args.maxseqlen
        orig_ctx_len = getattr(
            config, "max_position_embeddings", None
        )  # this value should be 4096 for LLaMA2 models
        if orig_ctx_len and context_size > orig_ctx_len:
            scaling_factor = float(math.ceil(context_size / orig_ctx_len))
            config.rope_scaling = {"type": "linear", "factor": scaling_factor}

        config._flash_attn_2_enabled = True

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        device_map={"": 0},
        trust_remote_code=True,
        cache_dir="/data/mgiles/shil6478/hf",
        torch_dtype=torch.bfloat16,
        use_safetensors=True,
    )
    print_mem()
    try:
        model.lm_head.cuda()
    except:
        pass

    if model_args.model_name_or_path == "togethercomputer/LLaMA-2-7B-32K":
        if config.vocab_size == 32001:
            model.resize_token_embeddings(32001)

    if model_args.load != "":
        model.load_state_dict(torch.load(model_args.load), strict=False)
        model.eval()

    _model = model.model
    _layers = _model.layers
    centroids_path = "/data/mgiles/shil6478/KVQuant/gradients/centroids/"

    if data_args.fisher not in [
        "fisher",
        "fisher_reloaded",
        "uniform",
        "fisher_kvquant",
    ]:
        raise ValueError("fisher not conventional, got " + data_args.fisher)

    dir_string = data_args.fisher + "/"
    nb_examples = data_args.num_examples
    t = model_args.t
    if model_args.outlier == "yes":
        outlier = True
    elif model_args.outlier == "no":
        outlier = False
    else:
        raise ValueError("outlier parameter not recognized")
    if model_args.all_layers == "yes":
        all_layers = True
    elif model_args.all_layers == "no":
        all_layers = False
    else:
        raise ValueError("all_layers parameter not recognized")
    model_name = model_args.model_name_or_path.rsplit("/", 1)[-1]

    ### Hybrid method for early layers
    if model_args.quant_type != "norm" or all_layers or model_args.size_centroids == 2:
        chan_layers = []
    elif model_args.size_centroids == 4:
        if model_name == "Llama-3.1-8B" or model_name == "Llama-3.2-3B":
            chan_layers = [0, 1, 2, 3]
        elif model_name == "Llama-3.2-1B":
            chan_layers = [0, 1, 2]
        else:
            chan_layers = [0, 1, 2, 3, 4]
    elif model_args.size_centroids == 8:
        if model_name == "Llama-3.1-8B":
            chan_layers = [0, 1, 2, 3, 4, 12]
        elif model_name == "Llama-3.2-3B":
            chan_layers = [0, 1, 2, 3, 4, 6, 7, 11]
        elif model_name == "Llama-3.2-1B":
            chan_layers = [0, 1, 2, 3, 4]
        else:
            chan_layers = []

    print("Loading layers...")
    print("Compressing layers ")
    cos_sin = CosSin()
    for layer in range(0, len(_layers)):
        if layer in chan_layers:
            file_name1 = (
                dir_string
                + "layer"
                + str(layer)
                + "size"
                + str(model_args.size_centroids)
                + "bit"
                + str(model_args.n_bits)
                + "nbexamples"
                + str(nb_examples)
                + model_name
                + ".safetensors"
            )
            file_name2 = (
                "nbexamples"
                + str(nb_examples)
                + "layer"
                + str(layer)
                + "means_stds_"
                + model_name
                + ".safetensors"
            )
            device = next(_layers[layer].parameters()).device
            _layers[layer].self_attn = PatchLlama31_8B(
                _model.config,
                centroids_path,
                file_name1,
                device,
                layer,
                file_name2,
                _layers[layer].self_attn,
                len(_layers),
                cos_sin,
                quant_type="chan",
                quant_type2=model_args.quant_type,
                t=t,
                outlier=outlier,
            )
    for layer in range(0, len(_layers)):
        if layer not in chan_layers:
            file_name1 = (
                dir_string
                + "layer"
                + str(layer)
                + "size"
                + str(model_args.size_centroids)
                + "bit"
                + str(model_args.n_bits)
                + "nbexamples"
                + str(nb_examples)
                + model_name
                + ".safetensors"
            )
            file_name2 = (
                "nbexamples"
                + str(nb_examples)
                + "layer"
                + str(layer)
                + "means_stds_"
                + model_name
                + ".safetensors"
            )
            device = next(_layers[layer].parameters()).device
            _layers[layer].self_attn = PatchLlama31_8B(
                _model.config,
                centroids_path,
                file_name1,
                device,
                layer,
                file_name2,
                _layers[layer].self_attn,
                len(_layers),
                cos_sin,
                quant_type=model_args.quant_type,
                quant_type2=model_args.quant_type,
                t=t,
                outlier=outlier,
            )
    print("Loading of layers finished")

    if is_perplexity:
        """Perplexity evaluation pipeline"""
        grads = {}
        acts = {}

        if data_args.trainset:
            loaded_examples = dataloader[:nb_examples]
        else:
            loaded_examples = dataloader[
                nb_examples : nb_examples + 4 * data_args.num_test_examples
            ]
        sum_losses = 0
        for k, data in tqdm(enumerate(loaded_examples)):
            # print_mem()
            data = data[0]
            x = data.to(device)
            model.zero_grad()
            with torch.no_grad():
                loss = model(input_ids=x, labels=x).loss.detach().item()
                sum_losses += loss
                del loss, x, data
            gc.collect()
            torch.cuda.empty_cache()
        perplexity = np.exp(sum_losses / len(loaded_examples))
        with open("results.txt", "a") as f:
            f.write("_______________________________________________" + "\n")
            f.write(
                "quant type : "
                + str(model_args.quant_type)
                + "/ outlier : "
                + model_args.outlier
                + "\n"
            )
            f.write(
                "model : "
                + model_args.model_name_or_path
                + "/ size : "
                + str(model_args.size_centroids)
                + "/ dataset : "
                + data_args.dataset
                + "\n"
            )
            f.write("Average Perplexity : " + str(perplexity) + "\n")
        with open("results.csv", "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "perplexity",
                    str(perplexity),
                    str(model_args.model_name_or_path).split("/")[-1],
                    str(model_args.quant_type),
                    model_args.outlier,
                    data_args.dataset,
                    str(model_args.size_centroids),
                ]
            )
        print(
            "model : "
            + model_args.model_name_or_path
            + "/ size : "
            + str(model_args.size_centroids)
            + "/ dataset : "
            + data_args.dataset
            + "\n"
        )
        print("quant_type : " + model_args.quant_type + "\n")
        print("Average Perplexity : " + str(perplexity) + "\n")
        if len(chan_layers) == 0:
            mses = np.zeros(
                (
                    len(_layers),
                    _layers[0].self_attn.num_key_value_heads * 2,
                    _layers[0].self_attn.head_dim,
                )
            )
            for layer in range(0, len(_layers)):
                key_mses = (
                    _layers[layer].self_attn.key_mses
                    / _layers[layer].self_attn.n_samples
                )
                value_mses = (
                    _layers[layer].self_attn.value_mses
                    / _layers[layer].self_attn.n_samples
                )
                mses[layer, : _layers[0].self_attn.num_key_value_heads] = (
                    key_mses.detach().cpu().numpy()
                )
                mses[layer, _layers[0].self_attn.num_key_value_heads :] = (
                    value_mses.detach().cpu().numpy()
                )
            mses_name = (
                "mse/"
                + str(model_args.quant_type)
                + "_"
                + str(model_args.outlier)
                + "_"
                + str(model_args.model_name_or_path.split("/")[-1])
                + "_"
                + str(model_args.size_centroids)
                + "_"
                + str(data_args.dataset)
                + ".npy"
            )
            np.save(mses_name, mses)
            print("MSES saved at " + mses_name)

    else:
        """Long context accuracy evaluation pipeline"""
        limit = 2048
        seq_lengths = [8192]
        batch_size = 4
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
        lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=batch_size)
        if "niah" in data_args.dataset or "ruler" in data_args.dataset:
            results = evaluator.simple_evaluate(
                model=lm,
                tasks=[data_args.dataset],
                batch_size=4,
                limit=limit,
                metadata={
                    "tokenizer": model_args.model_name_or_path,
                    "max_seq_lengths": seq_lengths,
                },
            )
            accuracy = [
                results["results"][data_args.dataset][str(seq_lengths[i]) + ",none"]
                for i in range(len(seq_lengths))
            ]
            uncertainties = [
                results["results"][data_args.dataset][
                    str(seq_lengths[i]) + "_stderr,none"
                ]
                for i in range(len(seq_lengths))
            ]  # _stderr,none
        else:
            results = evaluator.simple_evaluate(
                model=lm, tasks=[data_args.dataset], batch_size=4, limit=limit
            )
            accuracy = results["results"][data_args.dataset]["acc,none"]
        with open("results.txt", "a") as f:
            f.write("_______________________________________________\n")
            f.write(
                "quant type : "
                + str(model_args.quant_type)
                + "/outlier"
                + model_args.outlier
                + "/ dataset : "
                + data_args.dataset
                + "/ limit :"
                + str(limit)
                + "\n"
            )
            f.write(
                "model : "
                + model_args.model_name_or_path
                + "/ size : "
                + str(model_args.size_centroids)
                + "\n"
            )
            f.write("Accuracy : " + str(accuracy) + "\n")
            f.write("Uncertainties : " + str(uncertainties) + "\n")
        print(
            "model : "
            + model_args.model_name_or_path
            + " / size : "
            + str(model_args.size_centroids)
            + " / dataset : "
            + data_args.dataset
        )
        print("quant_type : " + model_args.quant_type)
        print("Average Accuracy : " + str(accuracy))

        ## Logging results
        with open("results.csv", "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "accuracy",
                    str(accuracy),
                    str(model_args.model_name_or_path).split("/")[-1],
                    str(model_args.quant_type),
                    model_args.outlier,
                    data_args.dataset,
                    str(model_args.size_centroids),
                ]
            )


if __name__ == "__main__":
    train()
