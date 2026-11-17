# E: B sampling with only the original functional gradient

`finetune_functional_only.py` prepares and manually runs an isolated five-epoch
ablation of the completed B experiment. Preparation reads B's plan, split and
five saved epoch orders, checks the original stage3 checkpoint hash, and copies
the split/orders into a new directory. It does not resume any B/D0/D1 model or
optimizer. All model parameters retain `requires_grad=True`, including the
structural branch on which Hf depends. No D1 loss or classification head is used.

All training settings are inherited from the completed B run: original
`stage3_best.pt`, original data, vocabularies and feature normalization, batch
size 32, seed 42, sampler seed 1000045 + epoch - 1, `circuit_cap=1024`, fresh Adam,
learning rate 3e-5, five epochs, original gradient clipping, and original LR
schedule. Sampling calls the same `sample_functional_pairs` with its independent
CPU generator. Each batch's circuit records and selected-pair-column hashes are
compared with B's existing `sampling.jsonl`; a mismatch stops E. The reference B
files are only read. Training never changes data files or source checkpoint.

The original total loss is

```
(3 * L_prob + 1 * L_rc + 2 * L_func + 2 * L_behavior)
/ sum(weights of originally active tasks)
```

In a usual batch all four tasks are active, so the effective coefficient of
`L_func` is `2/8 = 0.25`. E uses **`0.25 * L_func`**, not `L_func`. If B would
omit a task because its mask is empty, E uses that batch's original active
denominator. If the original functional term is absent, E has zero functional
gradient. There is no renormalization after disabling the other tasks.

`functional_only_objective` calls unchanged `loss_terms` under `no_grad` to
identify active original terms and record their reference values. It then
recomputes only the original functional term with autograd using
`functional_distance`, concatenation before normalization, `zero_normalize`
and L1. Probability/RC/behavior reference values do not contribute gradients;
there is no freezing or detachment anywhere inside the Hs-to-Hf path.

Per-batch logs explicitly report:

- `training_functional_objective`: scaled objective actually differentiated;
- `training_raw_functional_loss`: original functional term before weighting;
- `functional_coefficient_before/after` and `original_active_weight_sum`;
- `original_multitask_reference_loss`: detached **training sampled-pair**
  reference, not the validation loss;
- LR, gradient norm, finite flag, sampled counts/classes/duplicate rate,
  circuit records, and `B_pair_columns_match`.

Validation directly calls B's unchanged `evaluate_validation`: same original
validation circuits, batch size/order, complete original pairs, masks and
metrics, endpoint float64 conversion before chunked cosine, and a single
validation maximum-F1 threshold per epoch with grouped ties. It also runs once
before training. Training sampling is never applied in validation. Every
original metric is reported, including probability MAE, RC accuracy, TT-distance
MAE and all four behavior MAEs, macro/pooled/within-circuit AUC and the large-pair
circuit group. Logs compare metric deltas and deterioration against B at the
same epoch (MAE higher is worse; RC accuracy lower is worse).

E names the original validation total **`validation_original_multitask_total_loss`**
in JSON/checkpoints, separate from `training_functional_objective`. The old
`mean_loss` validation field is only renamed; its calculation is unchanged.
`best_loss.pt` uses the original validation total, `best_equiv.pt` uses pooled
validation F1, with baseline epoch 0 included. `last.pt`, per-epoch validation,
sampling logs and `history.json` are also saved. No test split is loaded.

The small self-check uses disposable models: real checkpoint weights on tiny
graphs to verify gradients, coefficients, actual optimizer update, inactive
heads unchanged and complete validation; then one real first B training batch
checks its exact sampled columns and does one disposable optimizer update.
It creates no formal E run or checkpoint. Unit tests also cover coefficients
when task masks are empty, exact functional gradients versus B, threshold/
validation reuse, overwrite refusal and the saving lifecycle.

The accompanying `D0_D1_existing_diagnostic.json` reads the old stored results
only. Those runs did **not** persist per-pair scores; checkpoints contain model
weights and scalar summaries. Consequently original-pair threshold reselection
and maximum recall at Precision >= 10%, 50%, 90% are unavailable from existing
files. They are not mislabeled “未达到”. No inference is run to recover them.
All retained D0/D1 results are labeled **训练电路诊断，不能当作验证或测试成绩**.

Prepared output directory:

```
/root/autodl-tmp/MyModel/Code/examples/exp/functional_pair_sampling_stage3_E_functional_only_20261109
```

Run manually (already prepared; do not repeat `--prepare`):

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
/root/miniconda3/envs/myModelEnv/bin/python -u \
  /root/autodl-tmp/MyModel/Code/examples/finetune_functional_only.py \
  --output /root/autodl-tmp/MyModel/Code/examples/exp/functional_pair_sampling_stage3_E_functional_only_20261109 \
  --run --device cuda
```

`[INFO]` output is flushed to the terminal and `console.log`. Training is
manually launched and monitored by the user. The script checks prepared code
hashes before starting and refuses to overwrite an already-started run.
