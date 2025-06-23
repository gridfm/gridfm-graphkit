# Loss Functions

### `Power Balance Equation Loss`

$$
\mathcal{L}_{\text{PBE}} = \frac{1}{N} \sum_{i=1}^N \left| (P_{G,i} - P_{D,i}) + j(Q_{G,i} - Q_{D,i}) - S_{\text{injection}, i} \right|
$$

::: gridfm_graphkit.utils.loss.PBELoss

---

### `Mean Squared Error Loss`

$$
\mathcal{L}_{\text{MSE}} = \frac{1}{N} \sum_{i=1}^N (y_i - \hat{y}_i)^2
$$

::: gridfm_graphkit.utils.loss.MSELoss

---

### `Masked Mean Squared Error Loss`

$$
\mathcal{L}_{\text{MaskedMSE}} = \frac{1}{|M|} \sum_{i \in M} (y_i - \hat{y}_i)^2
$$

::: gridfm_graphkit.utils.loss.MaskedMSELoss

---

### `Scaled Cosine Error Loss`

$$
\mathcal{L}_{\text{SCE}} = \frac{1}{N} \sum_{i=1}^N \left(1 - \frac{\hat{y}^T_i \cdot y_i}{\|\hat{y}_i\| \|y_i\|}\right)^\alpha \text{ , } \alpha \geq 1
$$

::: gridfm_graphkit.utils.loss.SCELoss

---

### `Mixed Loss`

$$
\mathcal{L}_{\text{Mixed}} = \sum_{m=1}^M w_m \cdot \mathcal{L}_m
$$

::: gridfm_graphkit.utils.loss.MixedLoss
