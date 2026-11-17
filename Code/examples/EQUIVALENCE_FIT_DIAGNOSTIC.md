# Fixed training-subset equivalence fitting diagnostic

训练子集拟合诊断，阈值在训练子集选择，不是验证或测试成绩。

`diagnose_equivalence_fit.py` is independent of pretraining and the A/B/C
experiments. Preparation creates `manifest.json` and a CPU `subset.pt` snapshot;
it does not start either 500-step run. The original dataset/checkpoint are read
without modification. The snapshot includes complete graphs, original pairs,
and selected source pair columns. Checkpoint vocabulary/scaling are inherited.

The default selection uses seed 42, train membership from the original
checkpoint, both positive/near-negative classes, and a configurable 20000-node
limit to bound the small diagnostic's cost. Sort the eligible circuits by
original valid-pair count and name, sample eight from the lowest quartile and
eight from the highest quartile, and shuffle their final saved order. An
independent CPU generator with seed 1000045 selects `min(128, n_pos, n_near)`
examples per class without replacement in each circuit. No epoch resampling.
The manifest records the population sizes, selection rule, local node pairs,
source columns, labels, masks, all 500 optimizer batches and both file hashes.

Each run makes 500 **optimizer steps**, cycling through four saved batches of
four circuits. Both runs independently load the same original stage3 weights
and use fresh Adam optimizers at 1e-4 (other Adam settings from the checkpoint).
Training computes float32 cosine, with normal autograd and clip norm 1.0.

- D0 exactly reuses the functional term's current standardized-distance L1:
  `l1_loss(zero_normalize(d), zero_normalize(tt))` over each training batch.
- D1 averages `.5*mean(d_positive) + .5*mean(relu(margin-d_negative))`
  within each circuit and then over circuits. `--margin` is configurable during
  preparation; default .02 is only a diagnostic trial value.
- Probability, RC and behavior losses are absent. No parameters are frozen.
  The structure branch remains active because Hf depends on Hs. The behavior
  branch and readout heads have no path into this loss, so they are unused.

At steps 0, 100, 250, 500 the model enters eval mode. The audited
`cosine_scores64` casts endpoints to float64 **before** chunked cosine. The
threshold maximizes pooled F1 on the selected balanced TRAIN pairs; complete
equal-score groups are included and ties in F1 prefer the larger threshold.
All original valid pairs of the same 16 circuits are evaluated at that **same**
threshold. No threshold is selected on those full pairs. Both scopes report
loss, P/R/F1, pooled/macro/within-circuit-weighted AUC, counts, confusion counts,
and positive/negative cosine-distance quantiles. Full-pair loss uses the same
batch grouping; D1 counts all original nonzero targets as negatives in that
additional evaluation. Evaluation returns the model to train mode.

Per-step logs include loss, actual LR, Hf gradient norm, structural/function/
embedding gradient norms and finite flags. Initial-versus-final parameter
changes are grouped by branch. The disposable self-check independently loads
both models and proves finite backward, structural/function parameter updates,
full-pair threshold reuse and restoration of train mode. It does not produce
formal run checkpoints.

Formal output: `D0/`, `D1/`, each containing `console.log`, `steps.jsonl`,
`history.json`, `metrics_step_*.json`, `run.json`, `last.pt`. The `--run both`
command executes D0 then D1 in one terminal and writes `comparison.json` with
initial/final metrics and gradient/update checks. It refuses to overwrite an
already-started run. A diagnostic success convention (F1 >= .95 and AUC >= .99
on fixed training pairs) is recorded in the manifest. If unmet, the conclusion
is only “本次预算下未成功”; this is no claim that the architecture cannot learn.

Prepare and check (already executed for the delivered experiment):

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 /root/miniconda3/envs/myModelEnv/bin/python -u \
  /root/autodl-tmp/MyModel/Code/examples/diagnose_equivalence_fit.py \
  --prepare --self-check \
  --data /root/autodl-tmp/MyModel/Code/examples/data/train \
  --checkpoint /root/autodl-tmp/MyModel/Code/examples/exp/triwave_gnn_20261026_144750_012230/stage3_best.pt \
  --output /root/autodl-tmp/MyModel/Code/examples/exp/equivalence_fit_diagnostic_20261109 \
  --lr 1e-4 --margin .02 --steps 500 --seed 42 --batch-size 4 --max-nodes 20000 --device cuda
```

Run manually, with continuously flushed `[INFO]` output and no agent monitoring:

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 /root/miniconda3/envs/myModelEnv/bin/python -u \
  /root/autodl-tmp/MyModel/Code/examples/diagnose_equivalence_fit.py \
  --output /root/autodl-tmp/MyModel/Code/examples/exp/equivalence_fit_diagnostic_20261109 \
  --run both --device cuda
```

Run parameters and original checkpoint/data locations come from the saved
manifest. To try another margin, prepare a **new** output directory first.
There is no validation/test evaluation, simulation, data generation, or
automatic transition to full training.
