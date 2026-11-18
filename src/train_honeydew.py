"""Single-GPU training entrypoint for the Honeydew multi-track Music Transformer."""
import argparse, json, random
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from lib.honeydew_midi import PAD, BOS, VOCAB_SIZE
from lib.model.transformer import MusicTransformer

class WindowDataset(Dataset):
    def __init__(self, samples, indices, seq_len):
        self.samples, self.indices, self.seq_len = samples, indices, seq_len
    def __len__(self):
        return len(self.indices)
    def __getitem__(self, item):
        seq = self.samples[self.indices[item]][:self.seq_len]
        out = torch.full((self.seq_len,), PAD, dtype=torch.long)
        if seq:
            out[:len(seq)] = torch.tensor(seq, dtype=torch.long)
        return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="honeydew_data.pt")
    ap.add_argument("--out_dir", default="honeydew_checkpoints")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--seq_len", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()
    random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Training device:", device)
    payload = torch.load(args.data, map_location="cpu", weights_only=False)
    samples, groups = payload["samples"], payload["groups"]
    if len(samples) < 2 or len(set(groups)) < 2:
        raise SystemExit("Need at least two MIDI source files to create a leak-free train/validation split.")

    unique = sorted(set(groups))
    random.shuffle(unique)
    val_count = max(1, int(len(unique) * 0.1))
    val_groups = set(unique[:val_count])
    train_ids = [i for i,g in enumerate(groups) if g not in val_groups]
    val_ids = [i for i,g in enumerate(groups) if g in val_groups]
    if not train_ids or not val_ids:
        raise SystemExit("Not enough source MIDI files for train/validation split.")

    train_ds = WindowDataset(samples, train_ids, args.seq_len)
    val_ds = WindowDataset(samples, val_ids, args.seq_len)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=args.workers, pin_memory=(device.type == "cuda"))
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers, pin_memory=(device.type == "cuda"))

    model = MusicTransformer(device, n_layers=6, d_model=512, dim_feedforward=2048,
                             num_heads=8, max_sequence=args.seq_len, rpr=False,
                             vocab_size=VOCAB_SIZE).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    best = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train(); total = 0.0; count = 0
        for batch in train_dl:
            batch = batch.to(device, non_blocking=True)
            x, target = batch[:, :-1], batch[:, 1:]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16,
                                enabled=(device.type == "cuda")):
                logits = model(x)
                loss = F.cross_entropy(logits.reshape(-1, VOCAB_SIZE),
                                       target.reshape(-1), ignore_index=PAD)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer); scaler.update()
            total += float(loss.item()); count += 1
        train_loss = total / max(1, count)

        model.eval(); val_total = 0.0; val_count = 0
        with torch.no_grad():
            for batch in val_dl:
                batch = batch.to(device, non_blocking=True)
                x, target = batch[:, :-1], batch[:, 1:]
                with torch.autocast(device_type=device.type, dtype=torch.float16,
                                    enabled=(device.type == "cuda")):
                    logits = model(x)
                    loss = F.cross_entropy(logits.reshape(-1, VOCAB_SIZE),
                                           target.reshape(-1), ignore_index=PAD)
                val_total += float(loss.item()); val_count += 1
        val_loss = val_total / max(1, val_count)
        print(f"epoch {epoch}/{args.epochs} train_loss={train_loss:.4f} val_loss={val_loss:.4f}")
        checkpoint = {"model": model.state_dict(), "model_config": {
            "n_layers": 6, "d_model": 512, "dim_feedforward": 2048,
            "num_heads": 8, "max_sequence": args.seq_len, "rpr": False,
            "vocab_size": VOCAB_SIZE}, "epoch": epoch,
            "train_loss": train_loss, "val_loss": val_loss}
        torch.save(checkpoint, out_dir / "latest.pt")
        if val_loss < best:
            best = val_loss
            torch.save(checkpoint, out_dir / "best.pt")
        with open(out_dir / "metrics.json", "w") as f:
            json.dump({"epoch": epoch, "train_loss": train_loss,
                       "val_loss": val_loss, "best_val_loss": best}, f, indent=2)

    print("Training complete. Best checkpoint:", out_dir / "best.pt")

if __name__ == "__main__":
    main()
