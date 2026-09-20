#!/usr/bin/env python
"""Train the internal-source conditional-flow GMC sampler."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
if not torch.cuda.is_available():
    torch.set_num_threads(1)
from torch.utils.data import DataLoader, TensorDataset

from gmc.mc_cell import generate_internal_dataset, load_internal_npz, save_npz
from gmc.model import BoundaryVelocityNet, ModelConfig, cfm_loss
from gmc.transforms import InternalTransform


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="")
    parser.add_argument("--generate-if-missing", action="store_true")
    parser.add_argument("--n-train", type=int, default=50_000)
    parser.add_argument("--n-val", type=int, default=5_000)
    parser.add_argument("--paper-preset", action="store_true", help="Use 9e5 train + 1e5 val + 100 epochs")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=4e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--hidden-dim", type=int, default=160)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--seed", type=int, default=4321)
    parser.add_argument("--out", type=str, default="outputs/internal_cfm.pt")
    args = parser.parse_args()

    if args.paper_preset:
        args.n_train = 900_000
        args.n_val = 100_000
        args.epochs = 100
        args.batch_size = 1024
        args.lr = 4e-4
        args.weight_decay = 1e-5

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")

    n_total = args.n_train + args.n_val
    if args.data:
        data_path = Path(args.data)
    else:
        data_path = Path(f"outputs/internal_raw_{n_total}.npz")

    if data_path.exists():
        batch = load_internal_npz(str(data_path))
        print(f"loaded {data_path}")
    else:
        if not args.generate_if_missing:
            raise FileNotFoundError(
                f"{data_path} not found. Run 08_generate_internal_data.py first or pass --generate-if-missing."
            )
        print(f"generating {n_total} internal MC samples ...")
        batch = generate_internal_dataset(n_total, seed=args.seed, progress_every=max(1, n_total // 20))
        data_path.parent.mkdir(parents=True, exist_ok=True)
        save_npz(str(data_path), batch)
        print(f"saved raw data to {data_path}")

    if batch.conditions.shape[0] < n_total:
        raise ValueError(f"dataset has {batch.conditions.shape[0]} samples, need {n_total}")

    conditions = batch.conditions[:n_total]
    targets = batch.targets[:n_total]
    c_train_raw, c_val_raw = conditions[: args.n_train], conditions[args.n_train :]
    y_train_raw, y_val_raw = targets[: args.n_train], targets[args.n_train :]

    transform = InternalTransform.fit(c_train_raw, y_train_raw)
    c_train = transform.encode_conditions(c_train_raw)
    y_train = transform.encode_targets(y_train_raw, c_train_raw)
    c_val = transform.encode_conditions(c_val_raw)
    y_val = transform.encode_targets(y_val_raw, c_val_raw)

    train_ds = TensorDataset(torch.from_numpy(c_train), torch.from_numpy(y_train))
    val_c = torch.from_numpy(c_val).to(device)
    val_y = torch.from_numpy(y_val).to(device)
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)

    config = ModelConfig(c_dim=6, y_dim=4, hidden_dim=args.hidden_dim, depth=args.depth)
    model = BoundaryVelocityNet(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.99), weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr,
        epochs=args.epochs,
        steps_per_epoch=len(loader),
        pct_start=0.15,
        anneal_strategy="cos",
    )

    best_val = float("inf")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for c_b, y_b in loader:
            c_b = c_b.to(device)
            y_b = y_b.to(device)
            loss = cfm_loss(model, c_b, y_b)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            running += float(loss.detach().cpu())
        train_loss = running / max(1, len(loader))

        model.eval()
        with torch.no_grad():
            if val_c.shape[0] > 50_000:
                idx = torch.randperm(val_c.shape[0], device=device)[:50_000]
                val_loss = cfm_loss(model, val_c[idx], val_y[idx]).item()
            else:
                val_loss = cfm_loss(model, val_c, val_y).item()
        print(
            f"epoch {epoch:04d}/{args.epochs} train_loss={train_loss:.6e} val_loss={val_loss:.6e} lr={scheduler.get_last_lr()[0]:.3e}",
            flush=True,
        )
        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "model_config": config.to_dict(),
                    "transform": transform.to_dict(),
                    "train_args": vars(args),
                    "best_val_loss": best_val,
                    "model_kind": "internal",
                },
                out,
            )
            with open(out.with_suffix(".json"), "w", encoding="utf-8") as f:
                json.dump({"best_val_loss": best_val, "train_args": vars(args)}, f, indent=2)
            print(f"saved best checkpoint to {out}")


if __name__ == "__main__":
    main()
