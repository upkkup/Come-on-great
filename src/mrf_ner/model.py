"""Feature-based BIO named entity recognition with Viterbi decoding.

The model scores a label sequence with a sum of *node* scores (token features
paired with a label) and *edge* scores (adjacent label pairs).  It is trained
from smoothed feature and transition counts, which keeps the implementation
dependency-free while still assigning a finite score to unseen observations.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from math import log
from typing import DefaultDict, Iterable, Sequence


class MarkovRandomFieldNER:
    """A first-order, feature-based NER model for BIO-tagged sentences.

    Parameters
    ----------
    smoothing:
        Additive smoothing applied to node-feature and transition counts.  It
        must be positive so unseen feature/label pairs remain decodable.
    """

    _START = "<START>"

    def __init__(self, smoothing: float = 1.0) -> None:
        if smoothing <= 0:
            raise ValueError("smoothing must be greater than zero")
        self.smoothing = smoothing
        self.labels: tuple[str, ...] = ()
        self._feature_counts: DefaultDict[str, Counter[str]] = defaultdict(Counter)
        self._label_feature_totals: Counter[str] = Counter()
        self._transition_counts: DefaultDict[str, Counter[str]] = defaultdict(Counter)
        self._transition_totals: Counter[str] = Counter()

    @staticmethod
    def _shape(token: str) -> str:
        """Return a compact orthographic shape, e.g. ``OpenAI42 -> XxXxdd``."""
        return "".join(
            "X" if char.isupper() else "x" if char.islower() else "d" if char.isdigit() else char
            for char in token
        )

    @classmethod
    def _features(cls, sentence: Sequence[str], index: int) -> tuple[str, ...]:
        """Extract observation features for one token and its local context."""
        token = sentence[index]
        lower = token.lower()
        previous = sentence[index - 1].lower() if index else "<BOS>"
        following = sentence[index + 1].lower() if index + 1 < len(sentence) else "<EOS>"
        return (
            f"word={lower}",
            f"shape={cls._shape(token)}",
            f"is_upper={token.isupper()}",
            f"is_title={token.istitle()}",
            f"has_digit={any(char.isdigit() for char in token)}",
            f"prefix1={lower[:1]}",
            f"prefix2={lower[:2]}",
            f"suffix1={lower[-1:]}",
            f"suffix2={lower[-2:]}",
            f"previous={previous}",
            f"next={following}",
        )

    @staticmethod
    def _split_bio(label: str) -> tuple[str, str | None]:
        if label == "O":
            return "O", None
        prefix, separator, entity_type = label.partition("-")
        if separator != "-" or prefix not in {"B", "I"} or not entity_type:
            raise ValueError(f"invalid BIO label: {label!r}")
        return prefix, entity_type

    @classmethod
    def _valid_transition(cls, previous: str, current: str) -> bool:
        """Whether ``previous -> current`` can occur in a legal BIO sequence."""
        current_prefix, current_type = cls._split_bio(current)
        if current_prefix != "I":
            return True
        if previous == cls._START:
            return False
        previous_prefix, previous_type = cls._split_bio(previous)
        return previous_prefix in {"B", "I"} and previous_type == current_type

    def fit(
        self, sentences: Iterable[Sequence[str]], labels: Iterable[Sequence[str]]
    ) -> "MarkovRandomFieldNER":
        """Fit smoothed node and edge potentials from aligned labeled sentences."""
        sentences = list(sentences)
        labels = list(labels)
        if len(sentences) != len(labels):
            raise ValueError("sentences and labels must have the same number of items")

        label_set: set[str] = set()
        for sentence, sequence in zip(sentences, labels):
            if len(sentence) != len(sequence):
                raise ValueError("every sentence must have the same length as its labels")
            previous = self._START
            for index, tag in enumerate(sequence):
                self._split_bio(tag)
                if not self._valid_transition(previous, tag):
                    raise ValueError(f"illegal BIO transition: {previous!r} -> {tag!r}")
                label_set.add(tag)
                for feature in self._features(sentence, index):
                    self._feature_counts[tag][feature] += 1
                    self._label_feature_totals[tag] += 1
                self._transition_counts[previous][tag] += 1
                self._transition_totals[previous] += 1
                previous = tag

        if not label_set:
            raise ValueError("training data must contain at least one labeled token")
        self.labels = tuple(sorted(label_set))
        return self

    def _node_score(self, sentence: Sequence[str], index: int, label: str) -> float:
        vocabulary_size = len(self._feature_counts[label]) + 1
        denominator = self._label_feature_totals[label] + self.smoothing * vocabulary_size
        return sum(
            log((self._feature_counts[label][feature] + self.smoothing) / denominator)
            for feature in self._features(sentence, index)
        )

    def _edge_score(self, previous: str, current: str) -> float:
        if not self._valid_transition(previous, current):
            return float("-inf")
        denominator = self._transition_totals[previous] + self.smoothing * len(self.labels)
        return log((self._transition_counts[previous][current] + self.smoothing) / denominator)

    def predict(self, sentence: Sequence[str]) -> list[str]:
        """Return the highest-scoring legal BIO sequence for ``sentence``."""
        if not self.labels:
            raise RuntimeError("fit must be called before predict")
        if not sentence:
            return []

        scores: list[dict[str, float]] = []
        backpointers: list[dict[str, str]] = []
        first_scores = {
            tag: self._edge_score(self._START, tag) + self._node_score(sentence, 0, tag)
            for tag in self.labels
        }
        scores.append(first_scores)
        backpointers.append({})

        for index in range(1, len(sentence)):
            current_scores: dict[str, float] = {}
            current_backpointers: dict[str, str] = {}
            for current in self.labels:
                previous, best_score = max(
                    (
                        (previous, scores[-1][previous] + self._edge_score(previous, current))
                        for previous in self.labels
                    ),
                    key=lambda item: item[1],
                )
                current_scores[current] = best_score + self._node_score(sentence, index, current)
                current_backpointers[current] = previous
            scores.append(current_scores)
            backpointers.append(current_backpointers)

        final_tag = max(self.labels, key=scores[-1].get)
        result = [final_tag]
        for index in range(len(sentence) - 1, 0, -1):
            result.append(backpointers[index][result[-1]])
        return list(reversed(result))
