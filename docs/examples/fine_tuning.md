# Fine-tuning an existing GridFM

Here we exploit the previously pre-trained reconstruction model to demonstrate the concept of fine-tuning. We exploit the power-flow problem, as a low-entry barrier example on how to fine-tune more complex downstream tasks in the future.
Thus, the overall workflow consists of:

1. As with the pre-training, the first step is to normalize the fine-tuning data and convert the network and power flow solution into a pytorch geometric graph representation

2. Data Loader then loads the data for fine-tuning

3. In the PF use-case, which is most closely related to the pre-training, we simply need to adjust the masking strategy to correspond to the PF problem, i.e. no longer random masking. For other use-cases, it may even be necessary to replace the decoder or add an additional head or decoder layer to the pre-trained autoencoder.

4. Then the model is trained to reconstruct the PF grid-state. As a loss, the standard "means square/absolute" error is used together with a physics informed loss, based on node-wise power balance equations (what comes in needs to get out...or be absorbed).

5. Once fine-tuned, we visualize fine-tuning performance and PF grid-state reconstruction


```python
# Load required libraries
# IBM GridFM library
from gridFM.datasets.powergrid import GridDatasetMem
from gridFM.datasets.data_normalization import BaseMVANormalizer
from gridFM.io.param_handler import NestedNamespace, get_transform, load_model, get_loss_function
from gridFM.training.trainer import Trainer
from gridFM.datasets.utils import split_dataset
from gridFM.datasets.transforms import AddPFMask
from gridFM.training.callbacks import EarlyStopper
from gridFM.training.plugins import MetricsTrackerPlugin
from gridFM.utils.loss import PBELoss

# Standard Libraries
import torch
from torch_geometric.loader import DataLoader
from torch.utils.data import Subset
import matplotlib.pyplot as plt
import os
import yaml
import warnings

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

```

## Load the Training Data and Create the Dataset

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

```python
# Select from which grid case file the simulated AC powerflow data should be used
data_dir = "../data/case30_ieee"

node_normalizer, edge_normalizer = BaseMVANormalizer(node_data=True), BaseMVANormalizer(node_data=False)

dataset = GridDatasetMem(
    root=data_dir,
    norm_method="baseMVAnorm",
    node_normalizer=node_normalizer,
    edge_normalizer=edge_normalizer,
    pe_dim=20,           # Dimension of positional encoding
    transform=AddPFMask()
)
```
## Split the Dataset for Training and Validation


```python
node_normalizer.to(device)
edge_normalizer.to(device)

train_dataset, val_dataset, _ = split_dataset(
    dataset, data_dir, val_ratio=0.1, test_ratio=0.1
)
```

## Create Pytorch Dataloaders for Training, Validation and Testing

```python
# Create DataLoaders with batches. The data-Loaders also take care of the masking for the powerflow problem formulation, the masking strategy in the configuration yaml needs to be set to "pf".
train_loader = DataLoader(
    train_dataset, batch_size=32, shuffle=True
)
val_loader = DataLoader(
    val_dataset, batch_size=32, shuffle=False
)
```

## Load the Model

```python
model = torch.load("../models/GridFM_v0_2_3.pth", weights_only=False, map_location=device).to(device)

# Select optimizer and learning rate scheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.0001,
)
# Adjust learning rate while training
scheduler = ReduceLROnPlateau(optimizer)

# This block only for compatibility with original code - does not do anything here
best_model_path = os.path.join("best_checkpoint.pth")
early_stopper = EarlyStopper(
    best_model_path, -1, 0
)
```

## Fine-Tune the model
```python
loss_fn = PBELoss()
# Plugin logs validation losses and saves to file for later use
log_val_loss_plugin = MetricsTrackerPlugin()
# Setup Trainer Instance
trainer = Trainer(
    model=model,
    optimizer=optimizer,
    device=device,
    loss_fn=loss_fn,
    early_stopper=early_stopper,
    train_dataloader=train_loader,
    val_dataloader=val_loader,
    lr_scheduler=scheduler,
    plugins=[log_val_loss_plugin],
)
trainer.train(epochs=15)
```

<p align="center">
  <img src="../figs/loss.png" alt="Loss"/>
  <br/>
</p>
