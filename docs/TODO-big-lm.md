# Big proxy LM: open items

Branch `feat/big-proxy-lm`. Done and tested: `--preset big` (settings in the README), the LM text pool
(`DataPipeline.lm_pool`, `--lm-text-bytes`), per-run result files with incremental download,
`--job-dir` / `--resume` / `--prepare-only`, creation of the remote `runs/` directory before held run
files are uploaded (`colab upload` does not create directories), compressed job files, and `--gpu none`
(a CPU runtime). A full-scale `--prepare-only` ran locally and a toy-sized job ran on a Colab CPU runtime,
including an interrupted run and its resume (numbers in the README).

Nothing has been trained on a GPU. Every item below needs the first L4 run.

- L4 memory: 3 concurrent 33M-parameter runs at batch 64 x 512 (bf16) are unmeasured. If one runs out
  of memory, the job fails with the traceback and the finished run files stay in the job dir: re-run
  with `--resume --concurrency 2` (or 1). Add gradient accumulation only if batch 64 alone does not fit.
- Concurrency vs sequential: the test compares with `rel=1e-5`, not bit-equality, because separate CPU
  processes may use different thread counts. On one GPU with the same seed the runs should match;
  check on the first run.
- Eval runs in fp32 (no autocast): correct, slower on the big model (estimated at minutes over the 33
  runs). If it matters, use bf16 autocast in `BpbEvaluator` with logits cast to fp32 (as training
  does) and check that BPB moves by < 0.001.
- Report: `scripts/gen_readme_tables.py` / `render_colab` have no `colab_big` table yet. Add it once
  `benchmarks/results/colab_big/extrinsic_bpb_results.json` exists.
- GPU time and cost in the README are estimates from FLOPs; replace them with the measured numbers.
