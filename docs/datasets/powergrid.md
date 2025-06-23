## `GridDatasetMem`

::: gridfm_graphkit.datasets.powergrid.GridDatasetMem

## Usage Example

```python
from gridfm_graphkit.datasets.data_normalization import IdentityNormalizer
from gridfm_graphkit.datasets.powergrid import GridDatasetMem

dataset = GridDatasetMem(
    root="./data",
    norm_method="identity",
    node_normalizer=IdentityNormalizer(),
    edge_normalizer=IdentityNormalizer(),
    pe_dim=10,
    mask_dim=6,
)
```
