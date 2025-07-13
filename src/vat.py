import torch
import torch.nn.functional as F
from torch import nn


def kl_divergence(
    p: torch.Tensor,
    q: torch.Tensor
) -> torch.Tensor:
    """
    Calcule la divergence de Kullback-Leibler entre deux distributions.

    Args:
        p (Tensor): Distribution cible.
        q (Tensor): Distribution perturbée.

    Returns:
        Tensor: Divergence KL par batch.
    """
    p = F.softmax(p, dim=1)
    q = F.softmax(q, dim=1)
    return torch.sum(p*torch.log(p/(q+1e-8)), dim=1)


def virtual_adversarial_loss(
    model: nn.Module,
    x: torch.Tensor,
    epsilon: float = 1e-2
) -> torch.Tensor:
    """
    Calcule la Virtual Adversarial Loss (VAT) pour un batch.

    Args:
        model (nn.Module): Le modèle à entraîner.
        x (Tensor): Batch d'entrée non étiqueté.
        epsilon (float): Perturbation maximale.

    Returns:
        Tensor: Perte VAT moyenne sur le batch.
    """
    x = x.clone().detach().requires_grad_(True)
    y = model(x)
    d = torch.randn_like(x)
    d = F.normalize(d, p=2.0)
    x_perturbed = x+epsilon*d
    y_perturbed = model(x_perturbed)
    loss = kl_divergence(y, y_perturbed).mean()
    return loss
