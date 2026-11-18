"""Prepare multi-track MIDI windows for Honeydew training.

Example:
  python prepare_honeydew.py --midi_dir /content/midi --out /content/honeydew_data.pt --target_windows 2000
"""
import argparse
from pathlib import Path
import random
import torch
from tqdm import tqdm
from lib.honeydew_midi import encode_midi, split_tokens, load_midi, PAD, BOS, EOS

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--midi_dir", required=True)
    ap.add_argument("--out", default="honeydew_data.pt")
    ap.add_argument("--seq_len", type=int, default=1024)
    ap.add_argument("--target_windows", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    random.seed(args.seed)
    files = sorted([*Path(args.midi_dir).rglob("*.mid"), *Path(args.midi_dir).rglob("*.midi")])
    if not files:
        raise SystemExit(f"No MIDI files found under {args.midi_dir}")

    samples, groups = [], []
    errors = []
    for path in tqdm(files, desc="Encoding MIDI"):
        try:
            tokens = encode_midi(load_midi(path))
            for chunk in split_tokens(tokens, args.seq_len):
                samples.append(chunk)
                groups.append(str(path.relative_to(args.midi_dir)))
        except Exception as exc:
            errors.append((str(path), repr(exc)))

    if not samples:
        raise SystemExit("No usable MIDI sequences found.")
    order = list(range(len(samples)))
    random.shuffle(order)
    if args.target_windows > 0:
        order = order[:args.target_windows]
    samples = [samples[i] for i in order]
    groups = [groups[i] for i in order]

    payload = {"samples": samples, "groups": groups, "seq_len": args.seq_len,
               "vocab_size": 772, "source_files": len(set(groups)),
               "errors": errors}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.out)
    print(f"Saved {len(samples)} windows from {len(set(groups))} source MIDI files to {args.out}")
    print(f"Skipped files with errors: {len(errors)}")

if __name__ == "__main__":
    main()
