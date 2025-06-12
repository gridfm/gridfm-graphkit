# Data Normalization



Normalization improves neural network training by ensuring features are well-scaled, preventing issues like exploding gradients and slow convergence. In power grids, where variables like voltage and power span wide ranges, normalization is essential.  
The `gridFM` package offers four methods:

- [`Min-Max Normalization`](#minmaxnormalizer)  
- [`Standardization (Z-score)`](#standardizer)  
- [`Identity (no normalization)`](#identitynormalizer)  
- [`BaseMVA Normalization`](#basemvanormalizer)

Each of these strategies implements a unified interface and can be used interchangeably depending on the learning task and data characteristics.  
Additionally, users can create their own custom normalizers by extending the base [`Normalizer`](#normalizer) class to suit specific needs.


## Available Normalizers

### `MinMaxNormalizer`

::: gridFM.datasets.data_normalization.MinMaxNormalizer 

### `Standardizer`

::: gridFM.datasets.data_normalization.Standardizer

### `BaseMVANormalizer`

::: gridFM.datasets.data_normalization.BaseMVANormalizer

### `IdentityNormalizer`

::: gridFM.datasets.data_normalization.IdentityNormalizer

## Usage Workflow

Example:

```python
from gridFM.normalization import MinMaxNormalizer
import torch

data = torch.randn(100, 5)  # Example tensor

normalizer = MinMaxNormalizer()
params = normalizer.fit(data)
normalized = normalizer.transform(data)
restored = normalizer.inverse_transform(normalized)
```

## Custom Normalizers

If you need a custom normalization strategy, you can create your own by subclassing the base interface:

### `Normalizer`

::: gridFM.datasets.data_normalization.Normalizer