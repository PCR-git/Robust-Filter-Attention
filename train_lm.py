"""
Train a language model (RFA or baseline) on WikiText-103 or BabyLM-2025.

    python train_lm.py --dataset wikitext103 --preset M3 --compile
    python train_lm.py --dataset babylm --preset B1 --batch-size 12
    python train_lm.py --dataset wikitext103 --preset M2 --set damping=0.5     # override any args field
    python train_lm.py --dataset wikitext103 --preset M3 --resume runs/wikitext103/.../resume.pt

Presets are defined in training/presets.py. Each run writes to <out-dir>/<dataset>/<name>__<date>/
(<name> defaults to <backbone>__<preset>):
    lm_epoch_N.pt   model weights after epoch N (the two most recent epochs; ~77 MB each)
    resume.pt       weights + optimizer/scheduler state of the latest epoch (deleted when the run finishes)
    metrics.txt     per-epoch train/validation loss, validation perplexity, time; test perplexity at the end
    training_progress_epoch_N.png
The default <out-dir> is $RFA_RUNS_DIR if set (e.g. a folder outside a synced source tree), else ./runs.
"""

import argparse
import ast
import datetime
import glob
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import torch

from training import single_epoch_lm, evaluate_lm
from training.lm_data import DATASETS, load_tokenizer, load_lm_datasets, make_dataloaders
from training.presets import PRESETS, make_args, build_language_model
from utils import count_parameters, seed_everything
from visualization import plot_training_progress_lm


def parse_overrides(pairs):
    out = {}
    for pair in pairs:
        key, _, value = pair.partition("=")
        try:
            out[key] = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            out[key] = value
    return out


def write_metrics(path, header, history, test=None):
    """Per-epoch train loss, validation loss/perplexity and time, plus the test result once known."""
    lines = [f"# {header}", f"{'epoch':>5} {'train_loss':>10} {'val_loss':>9} {'val_ppl':>9} {'time_s':>7}"]
    for i, (tl, vl, vp, t) in enumerate(zip(history["train_loss"], history["val_loss"], history["val_ppl"],
                                            history["epoch_times"])):
        lines.append(f"{i + 1:>5} {tl:>10.4f} {vl:>9.4f} {vp:>9.2f} {t:>7.0f}")
    if test is not None:
        lines.append(f"test_loss {test[0]:.4f}  test_ppl {test[1]:.2f}")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


DEFAULT_OUT_DIR = os.environ.get("RFA_RUNS_DIR", "runs")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", choices=DATASETS, required=True)
    p.add_argument("--preset", choices=sorted(PRESETS), required=True)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--compile", action="store_true", help="torch.compile the backbone (needs Triton; ~1.8x faster RFA)")
    p.add_argument("--grad-checkpointing", action="store_true", help="trade ~30%% speed for much less memory")
    p.add_argument("--small-embed-init", action="store_true", help="embedding init std 0.02 instead of the published N(0,1)")
    p.add_argument("--max-steps", type=int, default=None, help="limit train/eval batches per epoch (quick tests)")
    p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="override fields of args")
    p.add_argument("--resume", default=None, help="resume.pt of an unfinished run (continues in its directory)")
    p.add_argument("--name", default=None, help="run directory name (default <backbone>__<preset>)")
    p.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    p.add_argument("--data-dir", default="datasets")
    p.add_argument("--num-proc", type=int, default=4, help="processes for tokenization")
    p.add_argument("--seed", type=int, default=2025)
    p.add_argument("--gpu", type=int, default=0)
    cli = p.parse_args()

    seed_everything(cli.seed)
    device = torch.device(f"cuda:{cli.gpu}" if torch.cuda.is_available() else "cpu")
    args = make_args(cli.preset, device=device, batch_size=cli.batch_size, grad_checkpointing=cli.grad_checkpointing,
                     **parse_overrides(cli.set))

    # --- Data ---
    tokenizer = load_tokenizer()
    lm_datasets = load_lm_datasets(cli.dataset, tokenizer, block_size=args.seq_len, root=cli.data_dir, num_proc=cli.num_proc)
    loaders = make_dataloaders(lm_datasets, tokenizer, args.batch_size)
    if cli.max_steps:
        from itertools import islice
        loaders = {k: list(islice(v, cli.max_steps)) for k, v in loaders.items()}
    print({k: len(v) for k, v in loaders.items()}, "batches per split")

    # --- Model, optimizer, schedule ---
    model = build_language_model(args, embed_init_std=0.02 if cli.small_embed_init else None).to(device)
    print(f"{cli.preset}: {type(model.backbone).__name__}, {count_parameters(model):,} parameters")
    if cli.compile:
        model.backbone.compile()  # in place, so checkpoint keys are unchanged (first steps include compile time)

    optimizer = torch.optim.Adam(model.parameters(), lr=cli.lr, betas=(0.9, 0.999), eps=1e-8)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=cli.lr, epochs=cli.epochs, steps_per_epoch=len(loaders["train"]) + 2,
        pct_start=0.05, anneal_strategy="cos", div_factor=25.0, final_div_factor=1000.0)

    history = {"loss": [], "train_loss": [], "val_loss": [], "val_ppl": [], "epoch_times": []}
    start_epoch = 0
    if cli.resume:
        ckpt = torch.load(cli.resume, map_location=device, weights_only=False)
        if "optimizer_state_dict" not in ckpt:
            raise ValueError(f"{cli.resume} holds only model weights; resume from the run's resume.pt")
        if ckpt.get("total_epochs", cli.epochs) != cli.epochs:
            raise ValueError(f"Checkpoint is from a {ckpt['total_epochs']}-epoch run; resume with --epochs "
                             f"{ckpt['total_epochs']} (the one-cycle LR schedule cannot be extended).")
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        history, start_epoch = ckpt["history"], ckpt["epoch"] + 1
        print(f"Resumed from {cli.resume} at epoch {start_epoch}")

    if cli.resume:
        run_dir = os.path.dirname(os.path.abspath(cli.resume))
    else:
        name = cli.name or f"{type(model.backbone).__name__}__{cli.preset}"
        run_dir = os.path.join(cli.out_dir, cli.dataset, f"{name}__{datetime.date.today()}")
    os.makedirs(run_dir, exist_ok=True)
    metrics_path = os.path.join(run_dir, "metrics.txt")
    resume_path = os.path.join(run_dir, "resume.pt")
    header = f"{cli.dataset} {cli.preset} | python train_lm.py {' '.join(sys.argv[1:])}"

    # --- Training ---
    for epoch in range(start_epoch, cli.epochs):
        t0 = time.time()
        _, history = single_epoch_lm(model, loaders["train"], history, optimizer, args, scheduler=scheduler)
        history["epoch_times"].append(time.time() - t0)
        n_train = len(loaders["train"])
        history.setdefault("train_loss", []).append(sum(history["loss"][-n_train:]) / n_train)

        val_loss, val_ppl = evaluate_lm(model, loaders["validation"], args)
        history["val_loss"].append(val_loss)
        history["val_ppl"].append(val_ppl)
        print(f"Epoch {epoch + 1}/{cli.epochs} | train loss {history['train_loss'][-1]:.4f} | "
              f"val loss {val_loss:.4f} | val ppl {val_ppl:.2f} | {history['epoch_times'][-1]:.0f}s")

        ckpt = {"epoch": epoch, "total_epochs": cli.epochs, "model_state_dict": model.state_dict(),
                "history": history, "args": args}
        torch.save(ckpt, os.path.join(run_dir, f"lm_epoch_{epoch + 1}.pt"))
        for old in sorted(glob.glob(os.path.join(run_dir, "lm_epoch_*.pt")), key=os.path.getmtime)[:-2]:
            os.remove(old)  # keep the two most recent epochs
        if epoch + 1 < cli.epochs:  # written to a temporary file first, so a crash never leaves a broken resume.pt
            torch.save({**ckpt, "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict()}, resume_path + ".tmp")
            os.replace(resume_path + ".tmp", resume_path)
        elif os.path.exists(resume_path):
            os.remove(resume_path)
        plot_training_progress_lm(history, epoch, run_dir)
        write_metrics(metrics_path, header, history)

    test_loss, test_ppl = evaluate_lm(model, loaders["test"], args)
    print(f"Test | avg NLL {test_loss:.4f} | PPL {test_ppl:.2f}")
    write_metrics(metrics_path, header, history, test=(test_loss, test_ppl))


if __name__ == "__main__":
    main()
