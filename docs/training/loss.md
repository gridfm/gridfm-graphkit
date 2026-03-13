# Loss Functions

## Base Loss

::: gridfm_graphkit.training.loss.BaseLoss

---

## Mean Squared Error Loss

$$
\mathcal{L}_{\text{MSE}} = \frac{1}{N} \sum_{i=1}^N (y_i - \hat{y}_i)^2
$$

::: gridfm_graphkit.training.loss.MSELoss

---

## Masked Mean Squared Error Loss

$$
\mathcal{L}_{\text{MaskedMSE}} = \frac{1}{|M|} \sum_{i \in M} (y_i - \hat{y}_i)^2
$$

::: gridfm_graphkit.training.loss.MaskedMSELoss

---

## Masked Generator MSE Loss

::: gridfm_graphkit.training.loss.MaskedGenMSE

---

## Masked Bus MSE Loss

::: gridfm_graphkit.training.loss.MaskedBusMSE

---

## Mixed Loss

$$
\mathcal{L}_{\text{Mixed}} = \sum_{m=1}^M w_m \cdot \mathcal{L}_m
$$

::: gridfm_graphkit.training.loss.MixedLoss

---

## Layered Weighted Physics Loss

::: gridfm_graphkit.training.loss.LayeredWeightedPhysicsLoss

---

## Loss Per Dimension

::: gridfm_graphkit.training.loss.LossPerDim
