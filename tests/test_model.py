import pytest

from mrf_ner import MarkovRandomFieldNER


def test_predicts_aligned_legal_bio_sequence() -> None:
    model = MarkovRandomFieldNER().fit(
        [["Alice", "joined", "OpenAI"], ["Bob", "visited", "Paris"]],
        [["B-PER", "O", "B-ORG"], ["B-PER", "O", "B-LOC"]],
    )

    prediction = model.predict(["Alice", "joined", "OpenAI"])

    assert prediction == ["B-PER", "O", "B-ORG"]


def test_viterbi_never_starts_or_switches_an_i_entity() -> None:
    model = MarkovRandomFieldNER().fit(
        [["Acme", "Labs"], ["Jane", "Doe"]],
        [["B-ORG", "I-ORG"], ["B-PER", "I-PER"]],
    )

    prediction = model.predict(["Unknown", "Labs"])

    assert not prediction[0].startswith("I-")
    assert not (prediction[0] == "B-PER" and prediction[1] == "I-ORG")


def test_rejects_illegal_training_bio_sequence() -> None:
    with pytest.raises(ValueError, match="illegal BIO transition"):
        MarkovRandomFieldNER().fit([["Alice"]], [["I-PER"]])
