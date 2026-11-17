# Functional pair sampling experiments

Entry point: `finetune_functional_pairs.py`. Preparation and self-check do not
start a full training epoch. Only an explicit `--mode` starts an experiment.

The prepared experiments are under
`/root/autodl-tmp/MyModel/Code/examples/exp/functional_pair_sampling_stage3/`:

| Directory | Training functional pairs |
| --- | --- |
| `original` | All finite, mask-valid pairs |
| `circuit_cap` | Up to 1024 per circuit, uniform without replacement |
| `circuit_cap_stratified` | Same per-circuit budget, zero/near/far targets 1:1:2 |

All start from the original `stage3_best.pt`, with its architecture, vocabulary,
split, feature normalization, batch size 32 and seed 42. Adam starts with no
moments, learning rate 3e-5, and a new epoch counter (1 through 5). The original
Adam options and gradient clipping are retained. The original LR step interval
is 20, so there is no decay within these five epochs. The source checkpoint's
resume/completion state is never restored.

`epoch_orders.json` is identical for the three runs. A separate CPU generator
uses `seed + 1000003 + epoch - 1` for pair draws. Loader RNG is also separate.
Selection uses PyG node-to-circuit membership and rejects cross-circuit pairs.
Pair columns, labels and masks are indexed together; full graphs and all other
supervision remain unchanged. Functional zero-normalization is still over the
whole batch's selected pairs, exactly as in the existing loss, rather than per
circuit or chunk. Optional cosine chunking recomputes chunks during backward
and concatenates them before normalization. It changes memory use, not the loss
formula or gradient paths.

The stratified run uses the existing positive definition
`isclose(TT, 0, atol=1e-8, rtol=0)`. Near pairs are nonpositive-class distances
in `(0, .2]`, far pairs are distances `>= .8`. Other valid distances are retained
as a fourth stratum: reserve their rounded original fraction of each circuit's
budget, with at least one slot for each nonempty stratum. Divide the remaining
slots by 1:1:2, redistributing empty strata's weights. Short strata include every
available source pair before replacement draws. Sampling duplicate statistics
count unique source pair columns; no labels are manufactured.

Before the first optimizer step, the full original validation split is evaluated.
The result is shared as `baseline_stage3.json` across modes after matching its
checkpoint, data, validation split, batch size, device and evaluation code hashes.
Each epoch evaluates the same full original validation pairs, selects one pooled
validation F1 threshold with grouped ties, and reports pooled/macro/within-circuit
weighted AUC. The latter weights each both-class circuit's AUC by `n_pos*n_neg`
and excludes cross-circuit comparisons. It is not pooled AUC.

Validation also reports the unchanged full-batch stage-3 loss, probability MAE,
RC accuracy, TT cosine-distance MAE, and four behavior MAEs. Prediction MAEs use
the existing transformations and masks. Equivalence scores and reported TT MAE
use pre-cosine float64 conversion of the original float32 embeddings, in chunks.
The group with at least 10000 valid original pairs reports pooled FPR and recall
at the same global validation threshold. No test-split loader is constructed.

Each run writes:

- `plan.json`, `split.json`, `epoch_orders.json`: prepared settings and order.
- `run.json`: actual state, settings and completion status.
- `console.log`: stdout/stderr mirrored and flushed directly by the process.
- `sampling.jsonl`: every batch's LR, loss, gradients, class counts, duplicate
  rates, unique counts and per-circuit selected-column hashes.
- `baseline.json`, `val_epoch_N.json`, `history.json`: complete validation results
  and sampling summaries.
- `best_loss.pt`: lowest original full-validation loss.
- `best_equiv.pt`: highest pooled validation F1 (earlier epoch wins ties).
- `last.pt`: final epoch state. Baseline epoch 0 is eligible for both best files.

The modes can run concurrently if their combined peak GPU and host memory fit.
The shared initial baseline uses a process file lock: one process computes it,
the others wait and reuse it, then train independently. Serial execution is an
option when peak memory is uncertain. Existing started runs cannot be overwritten
or silently resumed. Full results and initial weights stay separate.
The process prints progress directly in the invoking terminal; for another
terminal, use `tail -f <mode-directory>/console.log`. There is no model-driven
progress monitoring or output forwarding service.

Preparation and self-check (already performed for the supplied source run):

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 /root/miniconda3/envs/myModelEnv/bin/python -u \
  /root/autodl-tmp/MyModel/Code/examples/finetune_functional_pairs.py \
  --prepare --self-check --device cuda
```

The self-check constructs only in-memory tiny graphs and temporary model copies;
it never regenerates/replaces a real dataset or saves updated source weights.
It exercises all three modes, actual stage-3 checkpoint weights, fresh Adam,
backward gradients, parameter updates and the full-pair validation code.
