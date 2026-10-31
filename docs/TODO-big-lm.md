# Big proxy LM: open items

Branch `feat/big-proxy-lm`. What is done: `--preset big` (8 layers, d_model 512, 8 heads, FFN 2048,
seq 512, batch 64 = 32,768 tokens/step, AdamW lr 6e-4, weight decay 0.1, warmup 2% + cosine, clip 1.0,
bf16 autocast on CUDA when supported, 100 MB budget, seeds 0/1/2, L4, 11 variants, output
`benchmarks/results/colab_big/`), `--concurrency N` (default 3 for big, 1 otherwise), per-run result
files `runs/<name>__s<seed>.json` written atomically on the VM, a job that resumes from finished run files,
incremental download of finished run files on every poll, and tests in `tests/e2e/test_bench_big_lm.py`. The small tier's defaults are unchanged (tested).

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

## 2. Resume across CLI invocations

Done: on every poll `ColabOrchestrator` downloads finished `runs/<name>__s<seed>.json` files into the
local job dir, and a new `run_job` on the same job dir uploads the held run files so the VM job skips
them (tested with the fake colab CLI in `test_finished_runs_are_downloaded_and_reused_after_a_drop`).

Open:

- The CLI (`colab/__main__.py`) builds the job in a `tempfile.mkdtemp` dir and deletes it in `finally`,
  so held run files do not survive a failed CLI invocation. Keep the job dir (e.g. under
  `<output_dir>/job/`) when the run fails and add a `--resume <job_dir>` that skips tokenizer training
  and encoding. Check that config.json matches before reusing run files.
- The real `colab upload` into `/content/runs/<file>` (a subdirectory) is untested; the fake CLI creates
  parent dirs. If the real CLI does not, upload flat and move the files in the control script.

## 3. Smaller items

- Eval runs in fp32 (no autocast); fine for correctness, slower on the big model. Consider bf16 autocast
  in `BpbEvaluator` with logits cast to fp32 (already done in training).
- The concurrency test compares with `rel=1e-5`, not bit-equality: separate CPU processes may use
  different thread counts. On one GPU with the same seed the runs should match; verify on the first run.
- Check L4 memory with 3 concurrent 33M-parameter runs at batch 64 x 512 (bf16). Lower `--concurrency`
  if it OOMs; consider gradient accumulation if batch 64 alone does not fit.
- Report generation (`scripts/gen_readme_tables.py`, `render_colab`) has no `colab_big` table yet. Add it
  after the first run.
