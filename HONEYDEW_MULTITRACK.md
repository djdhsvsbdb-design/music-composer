# Honeydew multi-track Music Transformer

This folder adds a separate multi-track MIDI path without changing the original single-piano Music Composer demo.

## What it does

- Encodes MIDI from all instrument tracks, including drums.
- Preserves GM program numbers and drum-channel markers.
- Trains a smaller Transformer configuration for a first Colab experiment.
- Splits validation data by source folder/file group to reduce leakage.
- Generates a multi-track `.mid` file for import into Honeydew.

This is symbolic MIDI generation, not direct audio synthesis. Honeydew renders instruments, effects, and mixing.

## Google Colab

Open [Honeydew_Multitrack_Training_Colab.ipynb](https://colab.research.google.com/github/djdhsvsbdb-design/music-composer/blob/honeydew-multitrack/src/Honeydew_Multitrack_Training_Colab.ipynb).

1. Select Runtime → Change runtime type → GPU.
2. Put extracted MIDI files in `MyDrive/Honeydew/MIDI`.
3. Run the notebook cells in order.
4. The best checkpoint is saved to `MyDrive/Honeydew/checkpoints/best.pt`.
5. The last cell generates and downloads `honeydew_generated.mid`.

## Local commands

From `music-composer/src`:

```bash
pip install torch pretty_midi tqdm numpy
python prepare_honeydew.py --midi_dir /path/to/midi --out honeydew_data.pt --target_windows 2000 --seq_len 1024
python train_honeydew.py --data honeydew_data.pt --out_dir honeydew_checkpoints --epochs 10 --batch_size 2 --seq_len 1024
python generate_honeydew.py --checkpoint honeydew_checkpoints/best.pt --out honeydew_generated.mid
```

## Dataset notes

The target of 2,000 means 2,000 training windows, not 2,000 unique songs. Keep stems from one composition in the same source folder so the validation split can keep them together. For a first test, use a small MIDI subset. Large dataset downloads may exceed free Colab storage and runtime limits.

Use only MIDI datasets whose licenses permit your intended use. Slakh2100 is a candidate for multi-track symbolic training; check the exact archive/version license and attribution requirements before redistribution or commercial use.

## Current integration boundary

The generated MIDI is imported into Honeydew using its existing MIDI import path. This branch does not yet add an in-app AI button, background inference service, or native Rust/Python model runtime. Those require a separate Honeydew application change and platform-specific packaging.
