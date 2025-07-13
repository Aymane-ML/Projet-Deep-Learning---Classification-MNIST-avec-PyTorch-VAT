import random
from typing import Tuple
from torchvision.transforms import Lambda
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset


def load_mnist_data(
    data_dir: str,
    labeled_fraction: float = 0.1,
    batch_size: int = 64
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Charge les données MNIST et les sépare en données
    étiquetées/non étiquetées/test.

    Args:
        data_dir (str): Répertoire où stocker/télécharger MNIST.
        labeled_fraction (float): Fraction des données à utiliser
        comme étiquetées. Defaults to 0.1.
        batch_size (int, optional): Taille des mini-lots pour tous
        les DataLoaders. Defaults to 64.

    Returns:
        _type_: - DataLoader des données étiquetées,
                - DataLoader des données non étiquetées,
                - DataLoader des données de test.
    """
    transform = transforms.Compose([
        Lambda(lambda x: x.convert("L")),
        transforms.ToTensor()
    ])
    full_train = datasets.MNIST(
        data_dir,
        train=True,
        download=True,
        transform=transform
    )
    test_dataset = datasets.MNIST(
        data_dir,
        train=False,
        download=True,
        transform=transform
    )
    indices = list(range(len(full_train)))
    random.shuffle(indices)
    split = int(len(full_train) * labeled_fraction)
    labeled_indices = indices[:split]
    unlabeled_indices = indices[split:]
    labeled_subset = Subset(full_train, labeled_indices)
    unlabeled_subset = Subset(full_train, unlabeled_indices)
    labeled_loader = DataLoader(
        labeled_subset,
        batch_size=batch_size,
        shuffle=True
    )
    unlabeled_loader = DataLoader(
        unlabeled_subset,
        batch_size=batch_size,
        shuffle=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )
    return labeled_loader, unlabeled_loader, test_loader
