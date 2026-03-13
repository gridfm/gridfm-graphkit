# Data Normalization



Normalization improves neural network training by ensuring features are well-scaled, preventing issues like exploding gradients and slow convergence. In power grids, where variables like voltage and power span wide ranges, normalization is essential.
The `gridfm-graphkit` package offers normalization methods based on the per-unit (p.u.) system:

- [`BaseMVA Normalization`](#heterodatamvanormalizer)
- [`Per-Sample BaseMVA Normalization`](#heterodatapersamplemvanormalizer)

Each of these strategies implements a unified interface and can be used interchangeably depending on the learning task and data characteristics.

> Users can create their own custom normalizers by extending the base [`Normalizer`](#normalizer) class to suit specific needs.


---

## Available Normalizers

### `Normalizer`

::: gridfm_graphkit.datasets.normalizers.Normalizer

---

### `HeteroDataMVANormalizer`

::: gridfm_graphkit.datasets.normalizers.HeteroDataMVANormalizer

---

### `HeteroDataPerSampleMVANormalizer`

::: gridfm_graphkit.datasets.normalizers.HeteroDataPerSampleMVANormalizer

---

## Usage Workflow

Example:

```python
from gridfm_graphkit.datasets.normalizers import HeteroDataMVANormalizer
from torch_geometric.data import HeteroData

# Create normalizer
normalizer = HeteroDataMVANormalizer(args)

# Fit on training data
params = normalizer.fit(data_path, scenario_ids)

# Transform data
normalizer.transform(hetero_data)

# Inverse transform to restore original scale
normalizer.inverse_transform(hetero_data)
```
