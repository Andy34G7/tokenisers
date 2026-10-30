"""
Markdown leaderboards rendered from benchmark result JSON ({"meta": ..., "models": ...}).

The CLIs write these next to their JSON, and scripts/gen_readme_tables.py puts the same text into
README.md, so README numbers always come from a result file. Variants whose tokenizer is
byte-identical to another (``meta["duplicates"]``) share one row.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence, Tuple


def _mb(n: Any) -> str:
    return f"{n / 1e6:.1f} MB" if isinstance(n, (int, float)) else "?"


def _summary(meta: Dict[str, Any]) -> str:
    data = "SYNTHETIC offline corpus" if meta.get("synthetic_data") else "real corpus"
    commit = (meta.get("git_commit") or "unknown")[:10] + (" (dirty)" if meta.get("git_dirty") else "")
    stale = (meta.get("native_extension") or {}).get("stale_sources")
    if stale:
        commit += f" (STALE native extension: built before {len(stale)} Rust source file(s) changed)"
    if meta.get("tokenizer_path"):  # tokka --tokenizer-path: nothing was trained by this run
        parts = [f"{data}: external tokenizer `{Path(meta['tokenizer_path']).name}`, not trained by this run "
                 "(its training data is unknown and was not checked against the evaluation split)"]
    else:
        parts = [f"requested V = {meta.get('requested_vocab_size', '?')}",
                 f"{data}: tokenizers trained on the train split ({_mb(meta.get('train_bytes'))})"]
    parts += [
        f"evaluated on the {'' if meta.get('tokenizer_path') else 'held-out '}{meta.get('eval_split', '?')} "
        f"split ({_mb(meta.get('eval_bytes'))})",
        f"commit `{commit}`",
        f"generated {meta.get('created_utc', '?')}",
    ]
    args = " ".join(meta.get("command", [])[3:])
    return "- " + "; ".join(parts) + f"\n- arguments: {f'`{args}`' if args else '(defaults)'}"


def _label(name: str, duplicates: Dict[str, str]) -> str:
    aliases = [d for d, c in duplicates.items() if c == name]
    return " = ".join(f"`{n}`" for n in [name, *aliases])


def _table(
    doc: Dict[str, Any],
    columns: Sequence[Tuple[str, Callable[[Dict[str, Any]], str]]],
    sort_key: Callable[[Dict[str, Any]], float],
    note: bool = True,
) -> List[str]:
    dups = doc["meta"].get("duplicates", {})
    rows = sorted(((n, m) for n, m in doc["models"].items() if n not in dups), key=lambda kv: sort_key(kv[1]))
    lines = [
        "| Tokenizer | " + " | ".join(h for h, _ in columns) + " |",
        "| :--- |" + " ---: |" * len(columns),
    ]
    for name, m in rows:
        lines.append(f"| {_label(name, dups)} | " + " | ".join(f(m) for _, f in columns) + " |")
    if dups and note:
        lines.append("")
        lines.append("Variants joined with `=` produced byte-identical tokenizers and are one result.")
    return lines


def render_tokka(doc: Dict[str, Any]) -> str:
    meta = doc["meta"]
    langs = [k for k in next(iter(doc["models"].values()), {}) if isinstance(k, str)
             and k not in ("macro_avg", "micro_avg", "unique_tokens", "vocab_size")]
    cols = [
        ("\\|V\\|", lambda m: str(m["vocab_size"])),
        ("Macro bytes/token (↑)", lambda m: f"{m['macro_avg']['bytes_per_token']:.3f}"),
        ("Micro bytes/token (↑)", lambda m: f"{m['micro_avg']['bytes_per_token']:.3f}"),
        ("Macro fertility (↓)", lambda m: f"{m['macro_avg']['subword_fertility']:.3f}"),
        ("Macro continued-token % (↓)", lambda m: f"{m['macro_avg']['continued_word_rate']:.2f}"),
        ("Unique tokens used", lambda m: str(m["unique_tokens"])),
    ]
    macro = meta.get("macro_languages", langs)
    rule = (f" with >= {meta['macro_min_docs']} held-out documents and >= {meta['macro_min_bytes']:,} bytes"
            if "macro_min_docs" in meta else "")
    head = [
        _summary(meta),
        f"- macro = unweighted mean over the {len(macro)} languages{rule} ({', '.join(macro)}); "
        f"micro = totals summed over all {len(langs)} languages (token-weighted); unique tokens = union over languages",
        "",
    ]
    sizes = ["", "Held-out text per language:", "",
             "| Language | Docs | Bytes | In macro |", "| :--- | ---: | ---: | :---: |"]
    for lang in langs:
        sizes.append(f"| {lang} | {meta.get('language_docs', {}).get(lang, '?')} | "
                     f"{meta.get('language_bytes', {}).get(lang, 0):,} | {'yes' if lang in macro else 'no'} |")
    return "\n".join(head + _table(doc, cols, lambda m: -m["macro_avg"]["bytes_per_token"]) + sizes)


def _f3(x: Any) -> str:
    """Three decimals; "n/a" for an undefined score (nothing was scored), never a fake 0.000."""
    return "n/a" if x is None else f"{x:.3f}"


def render_intrinsic(doc: Dict[str, Any]) -> str:
    meta = doc["meta"]
    cols = [
        ("\\|V\\|", lambda m: str(m["vocab_size"])),
        ("Bytes/token (↑)", lambda m: f"{m['bytes_per_token']:.3f}"),
        ("Fertility (↓)", lambda m: f"{m['fertility']:.3f}"),
        ("Rényi E1", lambda m: f"{m['renyi_e_1_0']:.3f}"),
        ("Rényi E2.5", lambda m: f"{m['renyi_e_2_5']:.3f}"),
        ("Morph F1 (↑)", lambda m: _f3(m["morph_f1"])),
        ("Ident F1 (↑)", lambda m: _f3(m["identifier_f1"])),
        ("Encode MB/s", lambda m: f"{m['encode_mb_per_sec']:.2f}"),
        ("Decode MB/s", lambda m: f"{m['decode_mb_per_sec']:.2f}"),
    ]
    first = next(iter(doc["models"].values()), {})
    morph = meta.get("morphology") or {}
    words = first.get("morph_words_by_lang", {})
    head = [
        _summary(meta),
        f"- morph F1: per-word boundary F1 on {first.get('morph_words', '?')} gold words "
        f"({morph.get('gold_source', 'SIGMORPHON 2022 test split')}), averaged per language, then over the "
        f"{len(words) or '?'} languages unweighted (per language below); identifier F1: "
        f"{first.get('identifiers', '?')} identifiers mined from held-out code; throughput: median of "
        f"{first.get('throughput_repeats', '?')} passes over {_mb(first.get('throughput_bytes'))} "
        f"({first.get('throughput_docs', '?')} documents, one encode call each) on {meta.get('host', {}).get('cpu', '?')}",
        "",
    ]
    lines = head + _table(doc, cols, lambda m: -m["renyi_e_2_5"])
    if words:
        by_lang = [(f"{lang} ({n:,} words)", (lambda lang: lambda m: _f3(m["morph_f1_by_lang"][lang]))(lang))
                   for lang, n in words.items()]
        by_lang.append(("Word-weighted", lambda m: _f3(m["morph_f1_word_weighted"])))
        lines += ["", "Morph F1 by language:", "", *_table(doc, by_lang, lambda m: -m["renyi_e_2_5"], note=False)]
    return "\n".join(lines)


def _pm(mean: float, sd: Any) -> str:
    """mean ± sd; just the mean when sd is undefined (a single seed)."""
    return f"{mean:.3f}" if sd is None else f"{mean:.3f} ± {sd:.3f}"


def render_colab(doc: Dict[str, Any]) -> str:
    meta = doc["meta"]
    domains = list(meta.get("val_domains", []))
    backend = meta.get("backend", {})
    if backend.get("device") == "cuda":
        where = f"Colab GPU `{backend.get('gpu')}`" if backend.get("kind") == "colab" else f"local GPU `{backend.get('gpu')}`"
    else:
        where = f"{backend.get('kind', '?')} CPU `{backend.get('cpu', '?')}`"
    fb = meta.get("fallback")
    cols = [("\\|V\\|", lambda m: str(m["vocab_size"])),
            ("Train bytes/token", lambda m: f"{m['train_bytes_per_token']:.3f}"),
            ("Tokens trained", lambda m: f"{m['tokens_consumed']:,}")]
    cols += [(f"{d} BPB (↓)", (lambda d: lambda m: _pm(m["val_domains"][d]["bpb"], m["val_domains"][d]["bpb_sd"]))(d))
             for d in domains]
    cols.append(("Macro BPB (↓)", lambda m: f"**{_pm(m['macro_val_bpb'], m['macro_val_bpb_sd'])}**"))
    seeds = meta.get("seeds") or []
    head = [
        _summary(meta),
        f"- backend: {where}" + (f" — FALLBACK after Colab failed: {fb.get('error', '')}" if fb else ""),
        f"- proxy LM: {meta.get('lm_architecture', 'architecture not recorded')}; "
        f"{', '.join(f'{k}={v}' for k, v in meta.get('lm', {}).items())}",
        f"- fixed budget {meta.get('byte_budget', 0):,} UTF-8 bytes of LM training text per tokenizer, drawn "
        "from the same train split the tokenizers were trained on (each document used once); seeds "
        f"{seeds} " + ("(mean ± sample sd)" if len(seeds) > 1 else "(single seed: no sd)"),
        "",
    ]
    return "\n".join(head + _table(doc, cols, lambda m: m["macro_val_bpb"]))


__all__ = ["render_tokka", "render_intrinsic", "render_colab"]
