import torch
import torch.nn as nn
from src.vat import kl_divergence, virtual_adversarial_loss


class DummyModel(nn.Module):
    """Un modèle factice pour tester VAT (renvoie des logits)."""

    def __init__(self, output_dim=10):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(28 * 28, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.flatten(x)
        return self.fc(x)


def test_kl_divergence_output_shape():
    """
    Vérifie que kl_divergence retourne un vecteur de batch_size éléments.
    """
    batch_size = 4
    logits1 = torch.randn(batch_size, 10)
    logits2 = torch.randn(batch_size, 10)
    kl = kl_divergence(logits1, logits2)
    assert kl.shape == (
        batch_size,
    ), "La KL divergence devrait renvoyer un vecteur de taille (batch_size,)"


def test_kl_divergence_is_zero_for_identical_inputs():
    """
    KL divergence entre deux distributions identiques ≈ 0.
    """
    logits = torch.randn(4, 10)
    kl = kl_divergence(logits, logits)
    assert torch.allclose(
        kl, torch.zeros_like(kl), atol=1e-6
    ), "KL(p, p) doit être proche de 0"


def test_virtual_adversarial_loss_returns_scalar():
    """
    Vérifie que virtual_adversarial_loss retourne un scalaire.
    """
    model = DummyModel()
    x = torch.randn(8, 1, 28, 28)
    loss = virtual_adversarial_loss(model, x)
    assert isinstance(loss, torch.Tensor), "La perte doit être un tensor"
    assert loss.ndim == 0, "La VAT loss doit être scalaire (tensor de dimension 0)"


def test_virtual_adversarial_loss_is_positive():
    """
    La VAT loss doit être positive (divergence entre deux prédictions).
    """
    model = DummyModel()
    x = torch.randn(8, 1, 28, 28)
    loss = virtual_adversarial_loss(model, x)
    assert (
        loss.item() > 0
    ), "La VAT loss doit être > 0 car les perturbations changent les sorties"


def test_virtual_adversarial_loss_different_epsilon():
    """
    Vérifie que la perte varie si epsilon varie.
    """
    model = DummyModel()
    x = torch.randn(8, 1, 28, 28)
    loss_small = virtual_adversarial_loss(model, x, epsilon=1e-3)
    loss_large = virtual_adversarial_loss(model, x, epsilon=1e-1)
    assert not (
        torch.allclose(loss_small, loss_large)
    ), "La VAT loss devrait varier avec epsilon"
