# Read-only task gradient diagnostic

`diagnose_task_gradients.py` compares original `stage3_best.pt` and the completed
E run's `best_equiv.pt` (epoch 4). It never creates an optimizer, calls
`optimizer.step`, clips gradients, loads validation/test graphs, or writes to
existing experiment files. Checkpoint hashes and model parameters/buffers are
checked before/after. Models use train mode with autograd; the reviewed current
architecture has no dropout or BatchNorm. All four losses use one forward graph
per batch. `autograd.grad(allow_unused=True)` preserves the distinction between
disconnected parameters (`None`) and connected zero gradients.

The fixed 32 batches are the first 32 complete batches from B's saved epoch-1
order, without outcome-based selection. Original batch size 32 and sampler seed
1000045 are reused. Both checkpoints load the same train graphs with the same
source normalization, seeds, `circuit_cap=1024` generator sequence and pair
column hashes. All circuit records are compared with B before each gradient
measurement. The two generated selected-column lists must match exactly.

The diagnostic retains the original formulas, including behavior subtask mean
and the original stage-3 **active** denominator. Usual coefficients are
prob=3/8, RC=1/8, func=2/8, behavior=2/8. It does not renormalize isolated task
gradients. The unchanged `loss_terms` provides a same-forward value check on
every batch; the sum of all four weighted gradients is additionally compared
with the direct total-loss gradient on batch 1 of each checkpoint. Unit tests
verify equality of values and gradients with original losses, including empty
task masks and chunked functional cosine calculations.

Module partition is built from **module object parameter identities** and then
mapped back to real `named_parameters`, with exhaustive, disjoint coverage:

- shared initialization: logic, role, pin embedding modules;
- structure path: `initialization.internal['s']`, both s aggregators and GRUs;
- function path: `initialization.internal['f']`, `pi_function`, both f aggregators
  and GRUs;
- behavior path and task heads: recorded separately for dependency/norm coverage.

These partitions reflect the reviewed equations: Hs propagation uses Hs; Hf
propagation receives [Hs,Hf]; Hb propagation receives [Hs,Hf,Hb]. Probability
reads Hf, RC reads Hs, behavior reads Hb. Actual per-task autograd connectivity
and nonzero-gradient batch counts for **every real parameter name** are saved.
Membership is not used as a substitute for an observed graph dependency.

For func vs prob/RC/behavior, cosine is computed on common autograd-connected
parameter tensors, including all elements and their zero entries. Module
results apply the same support rule. No shared tensor or a zero norm gives
`cosine=null`, `norm_ratio=null`, status “不适用”. Norms and dots are accumulated
on CPU float64 from original float32 gradients. Weighted norm ratio is
`other_norm/func_norm` on the same common tensors. Full module norms for each
task are also saved to expose magnitudes outside the intersection.

Func versus the other three weighted gradient sum uses the intersection of
func dependencies and the other tasks' dependency union. Func versus the full
multitask gradient uses all func-connected tensors. Gradient cancellation to
zero is also N/A. Summary statistics give applicable counts, negative fraction
(only over applicable batches), and q0/q5/q25/q50/q75/q95/q100 for cosines,
weighted norms and ratios. Batches have equal summary weight; no artificial
zero cosine is inserted for N/A batches.

Outputs: `plan.json`, `run.json`, flushed `console.log`, per-checkpoint
`batches.jsonl`, `result.json`, `selected_columns.json`, and `comparison.json`.
Per-batch results are flushed immediately; final JSON contains the parameter
dependency table and transparent descriptive candidate-ablation screen.
Candidates require >=75% negative cosine batches and median cosine < -0.1 in
the **same module at both checkpoints**. Median norm ratios outside [0.1,10]
are separately reported as scale differences, not automatically labeled
harmful. Unmet/mixed evidence is “不足以定位”, structural non-overlap is N/A.
The screening rule is a diagnostic convention, not a statistical causal test.

These are pre-optimizer gradients, not Adam updates or long-term effects.
No model architecture or loss is modified based on this diagnostic.

Actual prepared output:

```
/root/autodl-tmp/MyModel/Code/examples/exp/task_gradient_diagnostic_20261112
```

The run is authorized as a read-only diagnostic. The user monitors output;
the agent does not periodically poll long processes. To display continuous
output from the already-started process:

```bash
tail -n 20 -F /root/autodl-tmp/MyModel/Code/examples/exp/task_gradient_diagnostic_20261112/console.log
```

The final log line is `Completed task gradient diagnostic`. Ctrl+C in `tail`
only stops viewing the log. The runner refuses to overwrite a started run.
