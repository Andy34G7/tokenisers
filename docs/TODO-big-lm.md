# Big proxy LM: open items

Branch `feat/big-proxy-lm`. What is done: `--preset big` (8 layers, d_model 512, 8 heads, FFN 2048,
seq 512, batch 64 = 32,768 tokens/step, AdamW lr 6e-4, weight decay 0.1, warmup 2% + cosine, clip 1.0,
bf16 autocast on CUDA when supported, 100 MB budget, seeds 0/1/2, L4, 11 variants, output
`benchmarks/results/colab_big/`), `--concurrency N` (default 3 for big, 1 otherwise), per-run result
files `runs/<name>__s<seed>.json` written atomically on the VM, a job that resumes from finished run files,
and tests in `tests/e2e/test_bench_big_lm.py`. The small tier's defaults are unchanged (tested).

Nothing has been run on Colab.

## 1. LM text pool beyond the tokenizer train split (blocks the big run)

The 100 MB budget needs ~110 MB of LM text (budget + 10%, each document used once). The default
train split is ~52 MB, and the tokenizers must stay the ones trained on that 50 MB split. Today
`--preset big` exits before loading data with a message pointing here (tested).

Proposed design, in `python/tokenisers/data/pipeline.py`:

- Add an `lm` pool to `PreparedDataset`: documents pulled from the same source mix and in the same
  per-source proportions, *after* the documents that make up the default train/val/test splits, until it
  holds `lm_bytes` (e.g. 120 MB). The default splits must not change: the same seed and train budget
  must give byte-identical train/val/test (assert via the existing cache key / document hashes).
- Cache the pool under its own key (source revisions, seed, train budget, lm_bytes) so the default
  cache is untouched.
- De-duplicate the pool against val and test by sha256 of each document's UTF-8 bytes, and assert at
  runtime in the CLI that `sha256(lm docs) & sha256(val docs ∪ test docs)` is empty (reuse
  `common.assert_disjoint`, which the small tier already applies to train vs val).
- The LM text may include the tokenizer train split (as the small tier does) plus the pool; record in the
  results metadata how many LM bytes came from each.
- `_lm_train_docs` takes the combined list; BPB stays on the same val split as the small tier.
- Test: offline synthetic corpus, small `lm_bytes`; check default splits unchanged, pool disjoint from
  val/test by sha256, and the CLI assertion fires on an injected overlap.

## 2. Incremental download in the orchestrator

`ColabOrchestrator.run_job` (`python/tokenisers/benchmarks/colab/__init__.py`) still downloads only
`result.json` at the end. The VM side already writes `runs/<name>__s<seed>.json` per finished run.
To do:

- On every status poll, list `runs/` on the VM and download any run file not yet held locally into the
  local job dir's `runs/` (with the existing retried transfer).
- On a dropped session, re-upload the local `runs/` with the job, so the restarted VM job skips them
  (`lm.run_job` already skips existing run files; tested in `test_job_resumes_from_finished_run_files`).
- Test with the fake colab CLI in `tests/e2e/test_bench_proxy_lm.py` (`_fake_colab`): drop the session
  after one run file exists; check that file was downloaded and only the remaining runs retrain.

## 3. Smaller items

- Eval runs in fp32 (no autocast); fine for correctness, slower on the big model. Consider bf16 autocast
  in `BpbEvaluator` with logits cast to fp32 (already done in training).
- The concurrency test compares with `rel=1e-5`, not bit-equality: separate CPU processes may use
  different thread counts. On one GPU with the same seed the runs should match; verify on the first run.
- Check L4 memory with 3 concurrent 33M-parameter runs at batch 64 x 512 (bf16). Lower `--concurrency`
  if it OOMs; consider gradient accumulation if batch 64 alone does not fit.
- Report generation (`scripts/gen_readme_tables.py`, `render_colab`) has no `colab_big` table yet. Add it
  after the first run.
