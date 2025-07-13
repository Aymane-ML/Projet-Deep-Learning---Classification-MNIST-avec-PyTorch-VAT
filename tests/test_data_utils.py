from torch.utils.data import DataLoader
from src.data_utils import load_mnist_data


def test_load_mnist_data_returns_dataloaders() -> None:
    """
    Vérifie que load_mnist_data retourne bien trois objets DataLoader.
    """
    labeled_loader, unlabeled_loader, test_loader = load_mnist_data(
        data_dir="./data", labeled_fraction=0.1, batch_size=32
    )
    assert isinstance(labeled_loader, DataLoader)
    assert isinstance(unlabeled_loader, DataLoader)
    assert isinstance(test_loader, DataLoader)


def test_labeled_data_fraction_is_respected() -> None:
    """Vérifie que la fraction de données étiquetées respecte
    bien le paramètre labeled_fraction.
    """
    total_train_size = 60000
    labeled_fraction = 0.05
    batch_size = 64
    labeled_loader, unlabeled_loader, _ = load_mnist_data(
        data_dir="./data", labeled_fraction=labeled_fraction, batch_size=batch_size
    )
    labeled_dataset_size = len(labeled_loader.dataset)
    expected_labeled_size = int(total_train_size * labeled_fraction)
    assert abs(labeled_dataset_size - expected_labeled_size) <= 1


def test_data_shapes_are_correct() -> None:
    """Vérifie que les tensors dans les DataLoaders ont bien la forme
    (B, 1, 28, 28) pour les images.
    """
    labeled_loader, _, _ = load_mnist_data(
        "./data", labeled_fraction=0.1, batch_size=16
    )
    images, labels = next(iter(labeled_loader))
    assert images.shape[1:] == (1, 28, 28), "Les images doivent être au format MNIST"
    assert labels.ndim == 1, "Les labels doivent être unidimensionnels"


def test_batch_size_is_respected() -> None:
    """
    Vérifie que la taille des batchs correspond bien au paramètre batch_size.
    """
    batch_size = 20
    labeled_loader, _, _ = load_mnist_data(
        "./data", labeled_fraction=0.1, batch_size=batch_size
    )
    images, _ = next(iter(labeled_loader))
    assert images.size(0) == batch_size
