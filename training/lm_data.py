"""Tokenized, fixed-length language-modeling datasets (WikiText-103, BabyLM-2025)."""

import glob
import os
from itertools import chain

DATASETS = ("wikitext103", "babylm")


def load_tokenizer(path="./gpt2_tokenizer/"):
    from transformers import GPT2TokenizerFast
    tokenizer = GPT2TokenizerFast.from_pretrained(path, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _raw_datasets(name, root):
    from datasets import load_dataset
    if name == "wikitext103":
        d = os.path.join(root, "wikitext-103")
        files = {split: os.path.join(d, f"{prefix}-*.parquet")
                 for split, prefix in [("train", "train"), ("validation", "validation"), ("test", "test")]}
        return load_dataset("parquet", data_files=files)
    if name == "babylm":
        d = os.path.join(root, "BabyLM_2025")
        files = {"train": glob.glob(os.path.join(d, "train_100M", "*.train")),
                 "validation": glob.glob(os.path.join(d, "dev", "*.dev")),
                 "test": glob.glob(os.path.join(d, "test", "*.test"))}
        return load_dataset("text", data_files=files)
    raise ValueError(f"Unknown dataset {name!r}; choose from {DATASETS}")


def _tokenize(examples, tokenizer, skip_blank):
    texts = examples["text"]
    if skip_blank:
        texts = [t for t in texts if len(t.strip()) > 0]
    return tokenizer(texts)


def _group_texts(examples, block_size):
    """Concatenate a batch of tokenized texts and cut it into blocks of block_size tokens."""
    concatenated = {k: list(chain(*examples[k])) for k in examples.keys()}
    total_length = len(concatenated[list(examples.keys())[0]])
    if total_length >= block_size:
        total_length = (total_length // block_size) * block_size
    result = {k: [t[i:i + block_size] for i in range(0, total_length, block_size)] for k, t in concatenated.items()}
    result["labels"] = result["input_ids"].copy()
    return result


def load_lm_datasets(name, tokenizer, block_size=512, root="datasets", num_proc=4):
    """DatasetDict with train/validation/test splits of `block_size`-token blocks (as in the training notebooks)."""
    babylm = name == "babylm"
    raw = _raw_datasets(name, root)
    tokenized = raw.map(_tokenize, batched=True, num_proc=num_proc, remove_columns=["text"],
                        fn_kwargs={"tokenizer": tokenizer, "skip_blank": babylm})
    lm = tokenized.map(_group_texts, batched=True, num_proc=num_proc, fn_kwargs={"block_size": block_size})
    if babylm:  # BabyLM's short files leave some partial blocks
        lm = lm.filter(lambda x: len(x["input_ids"]) == block_size)
    return lm


class _Blocks:
    """Token blocks held in one [n_blocks, block_size] tensor; labels mask pad tokens with -100."""

    def __init__(self, input_ids, pad_token_id):
        self.input_ids = input_ids
        self.labels = input_ids.masked_fill(input_ids == pad_token_id, -100)

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i):
        return {"input_ids": self.input_ids[i], "labels": self.labels[i]}


def make_dataloaders(lm_datasets, tokenizer, batch_size):
    """
    Loaders of {"input_ids", "labels"} batches: the same batches, in the same shuffle order, as
    DataCollatorForLanguageModeling(mlm=False) on the Arrow datasets (used by the published runs),
    but read from memory, which saves ~10 ms of CPU work per training step.
    """
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    loaders = {}
    for split in ("train", "validation", "test"):
        column = lm_datasets[split].with_format("arrow")["input_ids"].combine_chunks()
        lengths = np.unique(column.value_lengths().to_numpy())
        if len(lengths) != 1:
            raise ValueError(f"{split}: token blocks have different lengths {lengths}")
        ids = column.flatten().to_numpy().astype(np.int64).reshape(len(column), lengths[0])
        blocks = _Blocks(torch.from_numpy(ids), tokenizer.pad_token_id)
        loaders[split] = DataLoader(blocks, batch_size=batch_size, shuffle=(split == "train"))
    return loaders
