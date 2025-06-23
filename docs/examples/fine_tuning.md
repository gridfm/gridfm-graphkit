# Fine-tuning an existing GridFM

Here we demonstrate how to leverage a previously pre-trained model to perform fine-tuning on downstream tasks. Specifically, we focus on the Power Flow (PF) problem, a fundamental task in power systems that involves computing the steady-state voltages and power injections in the grid.

The workflow consists of the following steps:

- Similar to pre-training, the first step is to normalize the data and convert the power grid into a PyTorch Geometric graph representation.

- A DataLoader then loads the data for fine-tuning.

- In the PF use case, which closely aligns with the pre-training setup, we adjust the masking strategy to match the PF problem, i.e. no longer using random masking. For other use cases, it may be necessary to modify the decoder or add additional heads or decoder layers to the pre-trained autoencoder.

-  The model is then trained to reconstruct the PF grid state. The loss function consists of a physics-informed loss based on node-wise power balance equations (ensuring power injected equals power consumed or absorbed).

$$
\mathcal{L}_{\text{PBE}} = \frac{1}{N} \sum_{i=1}^N \left| (P_{G,i} - P_{D,i}) + j(Q_{G,i} - Q_{D,i}) - S_{\text{injection}, i} \right|
$$

- Finally, we visualize fine-tuning performance.



```python
from gridfm_graphkit.datasets.powergrid import GridDatasetMem
from gridfm_graphkit.datasets.data_normalization import BaseMVANormalizer
from gridfm_graphkit.io.param_handler import NestedNamespace, get_transform, load_model, get_loss_function
from gridfm_graphkit.training.trainer import Trainer
from gridfm_graphkit.datasets.utils import split_dataset
from gridfm_graphkit.datasets.transforms import AddPFMask
from gridfm_graphkit.training.callbacks import EarlyStopper
from gridfm_graphkit.training.plugins import MetricsTrackerPlugin
from gridfm_graphkit.utils.loss import PBELoss

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

## Load the training data and create the dataset

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
## Split the dataset for training and validation


```python
node_normalizer.to(device)
edge_normalizer.to(device)

train_dataset, val_dataset, _ = split_dataset(
    dataset, data_dir, val_ratio=0.1, test_ratio=0.1
)
```

## Create Pytorch dataloaders for training, validation and testing

```python
# Create DataLoaders with batches. The data-Loaders also take care of the masking for the powerflow problem formulation, the masking strategy in the configuration yaml needs to be set to "pf".
train_loader = DataLoader(
    train_dataset, batch_size=32, shuffle=True
)
val_loader = DataLoader(
    val_dataset, batch_size=32, shuffle=False
)
```

## Load the model

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

## Fine-tune the model
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
