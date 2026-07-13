# DARE-Gate model skeleton

The package currently implements only the selected architecture:

1. bidirectional, direction-aware structural attention;
2. DeepGate2-style forward functional propagation;
3. fixed-probe behavior propagation with AND pair interactions;
4. optional reconvergence-source correction;
5. node-local cross-probe attention and a centered response decoder.

The external model input remains the AIG. Probe embeddings are learned model
parameters for the fixed probe suite; they are not per-sample workload inputs.

```python
from Code.daregate import DAREGateModel

model = DAREGateModel()
output = model(
    gate_type,       # [N], 0=PI, 1=AND, 2=NOT
    edge_index,      # [2, E], fanin -> gate
    forward_level,   # [N]
    backward_level=backward_level,
    rc_source=rc_source,  # [N], -1 when a node has no RC source
)
```

The principal shapes are:

- `output.structure`, `output.function`, `output.behavior`: `[N, 128]`
- `output.probe_behavior`: `[N, 8, 128]`
- `output.response`: `[N, 8, 2]`
- `output.signature`: `[N, 16]`

`output.response.sum(dim=1)` is zero up to floating-point precision by
construction. Reconvergence discovery belongs to graph preprocessing; the
model remains usable before that preprocessing is added by omitting
`rc_source` or setting every entry to `-1`.

