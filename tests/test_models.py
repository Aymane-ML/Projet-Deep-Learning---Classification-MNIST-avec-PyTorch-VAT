import torch
from src.models import (
    SimpleCNNWithDropout,
    PretrainedResNetForMNIST,
    PretrainedMobileNetForMNIST,
)


def test_simplecnn_output_shape() -> None:
    """Vérifie que SimpleCNNWithDropout retourne un tensor de la bonne forme."""
    model = SimpleCNNWithDropout()
    x = torch.randn(16, 1, 28, 28)  # batch de 16 images MNIST
    output = model(x)
    assert output.shape == (16, 10), "La sortie du modèle SimpleCNN n'est pas correcte"


def test_resnet_output_shape() -> None:
    """
    Vérifie que PretrainedResNetForMNIST retourne un tensor de la bonne forme.
    """
    model = PretrainedResNetForMNIST()
    x = torch.randn(8, 1, 28, 28)
    output = model(x)
    assert output.shape == (8, 10), "La sortie du modèle ResNet n'est pas correcte"


def test_mobilenet_output_shape() -> None:
    """Vérifie que PretrainedMobileNetForMNIST retourne
    un tensor de la bonne forme.
    """
    model = PretrainedMobileNetForMNIST()
    x = torch.randn(4, 1, 28, 28)
    output = model(x)
    assert output.shape == (4, 10), "La sortie du modèle MobileNet n'est pas correcte"


def test_model_forward_pass_no_error() -> None:
    """
    Vérifie qu'aucun modèle ne plante sur un forward pass simple.
    """
    models = [
        SimpleCNNWithDropout(),
        PretrainedResNetForMNIST(),
        PretrainedMobileNetForMNIST(),
    ]
    x = torch.randn(2, 1, 28, 28)
    for model in models:
        try:
            _ = model(x)
        except Exception as e:
            raise AssertionError(f"Erreur forward avec {model.__class__.__name__}: {e}")
