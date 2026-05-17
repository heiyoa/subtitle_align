from __future__ import annotations

import unittest

from subtitle_align.align import Word
from subtitle_align.gates import (
    DEFAULT_HALLUCINATION_LEXICON,
    apply_island_words_gate,
    apply_word_duration_gate,
)


class TestAntiHallucinationGates(unittest.TestCase):
    def test_word_duration_gate_removes_long_word(self) -> None:
        words = [
            Word(word=" hello", start=0.00, end=0.20, probability=0.9),
            Word(word=" working", start=1.00, end=3.80, probability=0.2),  # hallucinated long span
            Word(word=" there", start=4.00, end=4.20, probability=0.9),
        ]
        kept, removed = apply_word_duration_gate(
            words,
            max_word_dur=1.2,
            removed_word_placeholder="blank",
            hallucination_lexicon=set(DEFAULT_HALLUCINATION_LEXICON),
        )
        self.assertEqual(len(removed), 1)
        self.assertIn("word_duration_gate", removed[0].reason)
        self.assertTrue(all(w.word.strip() != "working" for w in kept if w.word))

    def test_island_gate_removes_island_cluster(self) -> None:
        # 1-2 words within 4s, surrounded by big gaps => remove.
        words = [
            Word(word=" hi", start=0.00, end=0.20, probability=0.9),
            Word(word=" there", start=0.25, end=0.45, probability=0.9),
            # big gap (silence)
            Word(word=" working", start=10.00, end=10.30, probability=0.3),
            # big gap
            Word(word=" ok", start=20.00, end=20.20, probability=0.9),
            Word(word=" bye", start=20.25, end=20.45, probability=0.9),
        ]
        kept, removed, warnings = apply_island_words_gate(
            words,
            hole_gap_sec=3.0,
            island_window_sec=4.0,
            island_max_words=2,
            removed_word_placeholder="blank",
            hallucination_lexicon=set(DEFAULT_HALLUCINATION_LEXICON),
        )
        # The single "working" word is an island cluster and should be removed.
        self.assertTrue(any("island_words_gate" in r.reason for r in removed))
        self.assertTrue(any("gate_island_words_removed" in w for w in warnings))
        self.assertTrue(all(w.word.strip() != "working" for w in kept if w.word))


if __name__ == "__main__":
    unittest.main()

