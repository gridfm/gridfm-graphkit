import torch.nn.functional as F


def masked_loss(pred, target, mask, reduction="mean"):
    return F.mse_loss(pred[mask], target[mask], reduction=reduction)


def mse_loss(pred, target, reduction="mean"):
    return F.mse_loss(pred, target, reduction=reduction)

def sce_loss(pred, target, mask=None, alpha=3):
    # Normalize the predictions and targets
    if mask is not None:
        pred = F.normalize(pred[mask], p=2, dim=-1)
        target = F.normalize(target[mask], p=2, dim=-1)
    else:
        pred = F.normalize(pred, p=2, dim=-1)
        target = F.normalize(target, p=2, dim=-1)
    
    # Compute the element-wise los
    loss = (1 - (pred * target).sum(dim=-1)).pow(alpha)
    
    return loss.mean()