"""Generate a multi-instrument MIDI arrangement from a Honeydew checkpoint."""
import argparse
from pathlib import Path
import torch
from lib.honeydew_midi import BOS, EOS, PAD, VOCAB_SIZE, decode_tokens
from lib.model.transformer import MusicTransformer

def sample_next(logits, temperature=1.0, top_k=40):
    logits = logits / max(temperature, 1e-4)
    # Reserved token ranges are not valid musical events.
    logits[288:384] = float("-inf")
    logits[512:640] = float("-inf")
    logits[PAD] = float("-inf")
    logits[BOS] = float("-inf")
    if top_k > 0:
        values, indices = torch.topk(logits, min(top_k, logits.numel()))
        probs = torch.softmax(values, dim=-1)
        return indices[torch.multinomial(probs, 1)]
    return torch.multinomial(torch.softmax(logits, dim=-1), 1)[0]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="honeydew_checkpoints/best.pt")
    ap.add_argument("--out", default="honeydew_generated.mid")
    ap.add_argument("--tokens", type=int, default=1024)
    ap.add_argument("--temperature", type=float, default=0.95)
    ap.add_argument("--top_k", type=int, default=40)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = ckpt["model_config"]
    model = MusicTransformer(device, n_layers=cfg["n_layers"], d_model=cfg["d_model"],
        dim_feedforward=cfg["dim_feedforward"], num_heads=cfg["num_heads"],
        max_sequence=cfg["max_sequence"], rpr=cfg["rpr"],
        vocab_size=cfg["vocab_size"]).to(device)
    model.load_state_dict(ckpt["model"]); model.eval()

    generated = [BOS]
    with torch.no_grad():
        for _ in range(min(args.tokens - 1, cfg["max_sequence"] - 1)):
            context = torch.tensor([generated[-cfg["max_sequence"]:]],
                                   dtype=torch.long, device=device)
            logits = model(context)[0, -1]
            nxt = int(sample_next(logits, args.temperature, args.top_k).item())
            generated.append(nxt)
            if nxt == EOS:
                break
    midi = decode_tokens(generated)
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    midi.write(str(out))
    print(f"Saved {len(midi.instruments)} MIDI tracks to {out}")

if __name__ == "__main__":
    main()
