"""Honeydew multi-track MIDI event codec.

Token layout (VOCAB_SIZE=772):
0..127 note-on pitch; 128..255 note-off pitch; 256..287 velocity;
288..383 reserved; 384..511 time shift (10 ms units, 1..1.28 s);
512..639 reserved; 640..767 GM program; 768 drum-channel marker;
769 EOS; 770 PAD; 771 BOS.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import numpy as np
import pretty_midi

NOTE_ON = 0
NOTE_OFF = 128
VELOCITY = 256
TIME_SHIFT = 384
PROGRAM = 640
DRUM = 768
EOS = 769
PAD = 770
BOS = 771
VOCAB_SIZE = 772
MAX_SHIFT = 128
SHIFT_SECONDS = 0.01


def encode_midi(midi: pretty_midi.PrettyMIDI) -> list[int]:
    events = []
    for inst_idx, inst in enumerate(midi.instruments):
        program = int(inst.program) if not inst.is_drum else 0
        for note in inst.notes:
            events.append((max(0.0, float(note.start)), 2, inst_idx, program,
                           bool(inst.is_drum), int(note.pitch), int(note.velocity)))
            events.append((max(0.0, float(note.end)), 0, inst_idx, program,
                           bool(inst.is_drum), int(note.pitch), 0))
    events.sort(key=lambda x: (x[0], x[1], x[2]))
    if not events:
        return [BOS, EOS]

    out = [BOS]
    current_time = 0.0
    current_inst = None
    for when, kind, inst_idx, program, is_drum, pitch, velocity in events:
        if inst_idx != current_inst:
            out.append(PROGRAM + max(0, min(127, program)))
            if is_drum:
                out.append(DRUM)
            current_inst = inst_idx

        delta = max(0.0, when - current_time)
        ticks = int(round(delta / SHIFT_SECONDS))
        while ticks > 0:
            step = min(ticks, MAX_SHIFT)
            out.append(TIME_SHIFT + step - 1)
            ticks -= step
        current_time = when

        if kind == 0:
            out.append(NOTE_OFF + max(0, min(127, pitch)))
        else:
            bucket = max(0, min(31, int(velocity) * 32 // 128))
            out.append(VELOCITY + bucket)
            out.append(NOTE_ON + max(0, min(127, pitch)))
    out.append(EOS)
    return out


def decode_tokens(tokens) -> pretty_midi.PrettyMIDI:
    midi = pretty_midi.PrettyMIDI()
    instruments = {}
    active = {}
    current_program = 0
    current_drum = False
    current_time = 0.0
    velocity = 96

    def get_instrument():
        key = (current_program, current_drum)
        if key not in instruments:
            inst = pretty_midi.Instrument(program=current_program, is_drum=current_drum)
            instruments[key] = inst
            midi.instruments.append(inst)
        return key, instruments[key]

    for raw in tokens:
        token = int(raw)
        if token in (PAD, BOS):
            continue
        if token == EOS:
            break
        if PROGRAM <= token < PROGRAM + 128:
            current_program = token - PROGRAM
            current_drum = False
        elif token == DRUM:
            current_drum = True
        elif TIME_SHIFT <= token < TIME_SHIFT + MAX_SHIFT:
            current_time += (token - TIME_SHIFT + 1) * SHIFT_SECONDS
        elif VELOCITY <= token < VELOCITY + 32:
            velocity = max(1, min(127, (token - VELOCITY + 1) * 4 - 2))
        elif NOTE_ON <= token < NOTE_ON + 128:
            pitch = token - NOTE_ON
            key, _ = get_instrument()
            active[(key, pitch)] = (velocity, current_time)
        elif NOTE_OFF <= token < NOTE_OFF + 128:
            pitch = token - NOTE_OFF
            key, inst = get_instrument()
            note = active.pop((key, pitch), None)
            if note is not None and current_time > note[1]:
                inst.notes.append(pretty_midi.Note(note[0], pitch, note[1], current_time))

    for (key, pitch), (vel, start) in active.items():
        end = max(start + 0.05, current_time)
        instruments[key].notes.append(pretty_midi.Note(vel, pitch, start, end))
    return midi


def split_tokens(tokens, seq_len=2048, min_tokens=128):
    """Non-overlapping training windows; always append EOS to a short tail."""
    body = list(tokens)
    windows = []
    for start in range(0, len(body), seq_len):
        chunk = body[start:start + seq_len]
        if len(chunk) < min_tokens:
            continue
        if chunk[-1] != EOS and len(chunk) < seq_len:
            chunk.append(EOS)
        windows.append(chunk)
    return windows


def load_midi(path: str | Path):
    return pretty_midi.PrettyMIDI(str(path))
