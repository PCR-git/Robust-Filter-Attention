"""
Length extrapolation: test perplexity of trained language models at context lengths beyond training.

    python eval_extrapolation.py --dataset wikitext103 runs/wikitext103/M2__2026-10-18 ...
    python eval_extrapolation.py --dataset wikitext103 --lengths 512 1024 2048 4096 8192 <run dirs>

Protocol of the paper (Test LMs notebook): the test split is cut into non-overlapping blocks of L tokens,
each block is one sequence (batch size 1), and the perplexity covers every next-token prediction in it.
Each model is rebuilt from the args stored in its checkpoint (the latest lm_epoch_N.pt in the run
directory). Results go to <run dir>/extrapolation.txt and are printed as a table.
"""

import argparse
import glob
import math
import os
import re
import time

import torch

from training.lm_data import load_tokenizer, _raw_datasets, _tokenize, _group_texts
from training.presets import build_language_model


def test_blocks(dataset, tokenizer, block_size, root, num_proc):
    """[n_blocks, block_size] token ids of the test split, grouped as in training."""
    babylm = dataset == "babylm"
    raw = _raw_datasets(dataset, root)["test"]
    tok = raw.map(_tokenize, batched=True, num_proc=num_proc, remove_columns=["text"],
                  fn_kwargs={"tokenizer": tokenizer, "skip_blank": babylm})
    lm = tok.map(_group_texts, batched=True, num_proc=num_proc, fn_kwargs={"block_size": block_size})
    ids = [x for x in lm["input_ids"] if len(x) == block_size]
    return torch.tensor(ids, dtype=torch.long)


def latest_checkpoint(run_dir):
    files = glob.glob(os.path.join(run_dir, "lm_epoch_*.pt"))
    if not files:
        raise FileNotFoundError(f"no lm_epoch_*.pt in {run_dir}")
    return max(files, key=lambda f: int(re.search(r"lm_epoch_(\d+)\.pt$", f).group(1)))


@torch.no_grad()
def perplexity(model, blocks, pad_token_id, device):
    total_nll, total_tokens = 0.0, 0
    for row in blocks:
        input_ids = row[None].to(device)
        labels = input_ids.masked_fill(input_ids == pad_token_id, -100)
        loss, _ = model.loss(input_ids, labels, causal=True)
        n = (labels[:, 1:] != -100).sum().item()
        total_nll += loss.item() * n
        total_tokens += n
    return math.exp(total_nll / total_tokens)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("runs", nargs="+", help="run directories written by train_lm.py")
    p.add_argument("--dataset", choices=("wikitext103", "babylm"), required=True)
    p.add_argument("--lengths", type=int, nargs="+", default=[512, 1024, 2048, 4096])
    p.add_argument("--data-dir", default="datasets")
    p.add_argument("--num-proc", type=int, default=4)
    p.add_argument("--gpu", type=int, default=0)
    cli = p.parse_args()

    device = torch.device(f"cuda:{cli.gpu}" if torch.cuda.is_available() else "cpu")
    tokenizer = load_tokenizer()
    blocks = {L: test_blocks(cli.dataset, tokenizer, L, cli.data_dir, cli.num_proc) for L in cli.lengths}
    print({L: tuple(b.shape) for L, b in blocks.items()}, "test blocks per length")

    table = {}
    for run_dir in cli.runs:
        ckpt_path = latest_checkpoint(run_dir)
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        args = ckpt["args"]
        args.device = device
        model = build_language_model(args).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        name = os.path.basename(os.path.normpath(run_dir)).split("__")[0]
        table[name] = {}
        lines = [f"# {ckpt_path} (preset {args.preset})", f"{'length':>6} {'test_ppl':>9} {'time_s':>7}"]
        for L in cli.lengths:
            t0 = time.time()
            ppl = perplexity(model, blocks[L], tokenizer.pad_token_id, device)
            table[name][L] = ppl
            lines.append(f"{L:>6} {ppl:>9.2f} {time.time() - t0:>7.0f}")
            print(f"{name:<16} L={L:<5} ppl {ppl:7.2f}  ({time.time() - t0:.0f}s)", flush=True)
        with open(os.path.join(run_dir, "extrapolation.txt"), "w") as f:
            f.write("\n".join(lines) + "\n")
        del model
        torch.cuda.empty_cache()

    print("\n" + f"{'model':<16}" + "".join(f"{'L=' + str(L):>9}" for L in cli.lengths))
    for name, row in table.items():
        print(f"{name:<16}" + "".join(f"{row[L]:>9.2f}" for L in cli.lengths))


if __name__ == "__main__":
    main()
