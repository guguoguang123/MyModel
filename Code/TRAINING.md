# DARE-Gate training manager

`DAREGateTrainer` owns optimization and training lifecycle state. The model
class remains responsible only for forward computation.

```python
from Code.daregate import (
    DAREGateBatch,
    DAREGateModel,
    DAREGateTrainer,
    ObjectiveConfig,
    TrainerConfig,
)

model = DAREGateModel()
trainer = DAREGateTrainer(
    model,
    trainer_config=TrainerConfig(
        learning_rate=1e-4,
        weight_decay=1e-5,
        checkpoint_dir="exp/daregate",
        early_stopping_patience=10,
    ),
    objective_config=ObjectiveConfig(
        reconstruction_weight=1.0,
        similarity_weight=0.1,
        decoupling_weight=0.01,
        auxiliary_warmup_epochs=5,
    ),
)

history = trainer.fit(train_loader, validation_loader, num_epochs=100)
```

Each loader must yield a `DAREGateBatch`, a mapping with the same field names,
or an attribute-based object containing at least:

- `gate_type`: `[N]`
- `edge_index`: `[2, E]`
- `forward_level`: `[N]`
- `signature`: `[N, 16]`

Optional fields are `backward_level`, `rc_source`, `node_features`,
`batch_index`, and `names`.

To resume full optimization state:

```python
trainer.load_checkpoint("exp/daregate/model_last.pt")
trainer.fit(train_loader, validation_loader, num_epochs=100)
```

The checkpoint contains model, optimizer, scheduler, AMP scaler, current
epoch, best metric, early-stopping state, training history, and all three
configuration dictionaries. Files are written atomically.

