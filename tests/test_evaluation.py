import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from src.evaluation import evaluate, evaluate_metrics, visualize_errors


class DummyModel(nn.Module):
    """Modèle factice pour tester les fonctions d'évaluation.
    Simule un classifieur MNIST avec une couche linéaire simple.
    """

    def __init__(self) -> None:
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(28 * 28, 10)  # 10 classes (MNIST)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.flatten(x)
        return self.fc(x)


def get_fake_mnist_loader(num_samples: int = 32) -> DataLoader:
    """
    Crée un DataLoader factice avec des données aléatoires au format MNIST.

    Args:
        num_samples (int): Nombre d’échantillons à générer. Default = 32.

    Returns:
        DataLoader: DataLoader contenant des images et labels simulés.
    """
    x_fake = torch.randn(num_samples, 1, 28, 28)
    y_fake = torch.randint(0, 10, (num_samples,))
    dataset = TensorDataset(x_fake, y_fake)
    return DataLoader(dataset, batch_size=8)


def test_evaluate_prints_accuracy(capfd) -> None:
    """
    Teste si evaluate(...) affiche bien un texte contenant la précision.
    """
    model = DummyModel()
    test_loader = get_fake_mnist_loader()
    evaluate(model, test_loader)
    out, _ = capfd.readouterr()
    assert "Précision sur les données de test" in out


def test_evaluate_metrics_output() -> None:
    """
    Teste si evaluate_metrics(...) retourne une matrice 10x10
    et un rapport de classification sous forme de string.
    """
    model = DummyModel()
    test_loader = get_fake_mnist_loader()
    conf_matrix, report = evaluate_metrics(model, test_loader)
    assert conf_matrix.shape == (10, 10)
    assert isinstance(report, str)
    assert "precision" in report.lower()


def test_visualize_errors_runs_without_crashing() -> None:
    """
    Vérifie que visualize_errors(...) s'exécute sans erreur,
    même avec des données et prédictions aléatoires.
    """
    model = DummyModel()
    test_loader = get_fake_mnist_loader()
    try:
        visualize_errors(model, test_loader, num_errors=5)
    except Exception as e:
        assert False, f"visualize_errors a levé une erreur : {e}"
