"""Smoke tests for Honeydew's multi-track MIDI codec."""
import unittest

import pretty_midi

from lib.honeydew_midi import BOS, EOS, VOCAB_SIZE, decode_tokens, encode_midi, split_tokens


class HoneydewMidiTests(unittest.TestCase):
    def make_song(self):
        midi = pretty_midi.PrettyMIDI(initial_tempo=120)
        piano = pretty_midi.Instrument(program=0)
        piano.notes.append(pretty_midi.Note(velocity=96, pitch=60, start=0.0, end=0.5))
        bass = pretty_midi.Instrument(program=33)
        bass.notes.append(pretty_midi.Note(velocity=80, pitch=36, start=0.25, end=0.75))
        drums = pretty_midi.Instrument(program=0, is_drum=True)
        drums.notes.append(pretty_midi.Note(velocity=110, pitch=36, start=0.0, end=0.1))
        midi.instruments.extend([piano, bass, drums])
        return midi

    def test_encode_decode_keeps_multiple_instruments_and_drums(self):
        encoded = encode_midi(self.make_song())
        decoded = decode_tokens(encoded)
        self.assertEqual(len(decoded.instruments), 3)
        self.assertEqual(sum(len(i.notes) for i in decoded.instruments), 3)
        self.assertTrue(any(i.is_drum for i in decoded.instruments))
        self.assertTrue(all(0 <= token < VOCAB_SIZE for token in encoded))

    def test_windows_start_with_bos_and_fit_sequence_limit(self):
        source = [BOS] + [256 + (i % 32) for i in range(500)] + [EOS]
        windows = split_tokens(source, seq_len=64, min_tokens=8)
        self.assertGreater(len(windows), 1)
        self.assertTrue(all(window[0] == BOS for window in windows))
        self.assertTrue(all(len(window) <= 64 for window in windows))
        self.assertEqual(windows[-1][-1], EOS)

    def test_too_small_sequence_length_is_rejected(self):
        with self.assertRaises(ValueError):
            split_tokens([BOS, EOS], seq_len=2)


if __name__ == "__main__":
    unittest.main()
