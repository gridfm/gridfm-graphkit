# Data Normalization



Normalization improves neural network training by ensuring features are well-scaled, preventing issues like exploding gradients and slow convergence. In power grids, where variables like voltage and power span wide ranges, normalization is essential.
The `gridfm-graphkit` package offers four methods:

- [`Min-Max Normalization`](#minmaxnormalizer)
- [`Standardization (Z-score)`](#standardizer)
- [`Identity (no normalization)`](#identitynormalizer)
- [`BaseMVA Normalization`](#basemvanormalizer)

Each of these strategies implements a unified interface and can be used interchangeably depending on the learning task and data characteristics.

> Users can create their own custom normalizers by extending the base [`Normalizer`](#normalizer) class to suit specific needs.


---

## Available Normalizers

### `Normalizer`

::: gridfm_graphkit.datasets.normalizers.Normalizer

---

### `MinMaxNormalizer`

::: gridfm_graphkit.datasets.normalizers.MinMaxNormalizer

---

### `Standardizer`

::: gridfm_graphkit.datasets.normalizers.Standardizer

---

### `BaseMVANormalizer`

::: gridfm_graphkit.datasets.normalizers.BaseMVANormalizer

---

### `IdentityNormalizer`

::: gridfm_graphkit.datasets.normalizers.IdentityNormalizer

---

## Usage Workflow

Example:

```python
from gridfm_graphkit.datasets.normalizers import MinMaxNormalizer
import torch

data = torch.randn(100, 5)  # Example tensor

normalizer = MinMaxNormalizer(node_data=True,args=None)
params = normalizer.fit(data)
normalized = normalizer.transform(data)
restored = normalizer.inverse_transform(normalized)
```
