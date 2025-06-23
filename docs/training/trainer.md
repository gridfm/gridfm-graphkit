# Trainer

A flexible, modular training loop designed for GNN models in the GridFM framework.
Handles training, validation, early stopping, learning rate scheduling, and plugin callbacks.

---

## `Trainer`

::: gridfm_graphkit.training.trainer.Trainer

---

## Usage Example

```python
from gridfm_graphkit.training.trainer import Trainer
from gridfm_graphkit.training.callbacks import EarlyStopper
from gridfm_graphkit.training.plugins import MLflowLoggerPlugin

trainer = Trainer(
    model=model,
    optimizer=optimizer,
    device=device,
    loss_fn=loss_function,
    early_stopper=EarlyStopper(save_path),
    train_dataloader=train_loader,
    val_dataloader=val_loader,
    lr_scheduler=scheduler,
    plugins=[MLflowLoggerPlugin()]
)

trainer.train(start_epoch=0, epochs=100)
```
