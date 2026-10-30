"""Unigram HF export stays id-exact where Viterbi ties are decided by float rounding.

Hugging Face ``tokenizers`` parses JSON numbers without serde_json's ``float_roundtrip``, so a
17-digit log-prob can load 1 ULP off. A run of one repeated character ties exactly between
permutations of the same tokens ("~~|~" vs "~|~~"), and those ULPs then pick a different
segmentation than native. Native Unigram models therefore keep only scores HF parses exactly."""

import json
import random

import tokenizers

from tokenisers import _core

CORPUS = [f"x = a ~ b == c ~~ d ~~~ e {i}" for i in range(60)]


def _template(tmp_path):
    _core.train_unigram(CORPUS, 300).export_hf_json(str(tmp_path / "template.json"))
    return json.loads((tmp_path / "template.json").read_text(encoding="utf-8"))


def _load(tmp_path, data):
    src = tmp_path / "src.json"
    src.write_text(json.dumps(data), encoding="utf-8")
    return _core.PyUnigramModel.from_hf_json(str(src))


def _export(tmp_path, model):
    path = tmp_path / "exported" / "tokenizer.json"
    model.export_hf_json(str(path))
    return path, tokenizers.Tokenizer.from_file(str(path))


def test_hf_reads_every_exported_score_exactly(tmp_path):
    data = _template(tmp_path)
    rng = random.Random(0)
    for entry in data["model"]["vocab"]:
        entry[1] = -rng.uniform(0.0, 40.0)  # ~14% of these load 1 ULP off as written
    for model in [_load(tmp_path, data), _core.train_unigram(CORPUS, 300)]:
        path, hf = _export(tmp_path, model)
        ours = json.loads(path.read_text(encoding="utf-8"))["model"]["vocab"]
        theirs = json.loads(hf.to_str())["model"]["vocab"]
        assert [a for a, b in zip(ours, theirs) if a != b] == []


def test_runs_of_one_character_encode_like_hf(tmp_path):
    data = _template(tmp_path)
    vocab = data["model"]["vocab"][:256]
    for entry in vocab:
        entry[1] = -30.0
    # Found by search: HF reads "~~~" as ...294, so "~" * 7 was [~~, ~~~, ~~] in HF and
    # [~~, ~~, ~~~] natively (both 2*s(~~) + s(~~~) in exact arithmetic).
    vocab[ord("~")][1] = -18.225694237006703
    data["model"]["vocab"] = vocab + [["~~", -2.550619694603964], ["~~~", -2.4580254978822946]]
    model = _load(tmp_path, data)
    _, hf = _export(tmp_path, model)
    for n in range(1, 40):
        text = "~" * n
        assert model.encode(text) == hf.encode(text, add_special_tokens=False).ids, n
