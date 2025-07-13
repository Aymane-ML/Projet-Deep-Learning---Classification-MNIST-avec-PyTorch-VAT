from typing import List, Tuple, Any, Iterable
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter


def show_one_image_per_label(dataset: Iterable[Tuple[Any, int]]) -> None:
    """Affiche une image pour chaque étiquette (0 à 9) à partir d’un dataset.
    Utile pour valider visuellement la diversité des classes dans
    l'ensemble de données.

    Args:
        dataset (Iterable[Tuple[Any, int]]):
            Itérable contenant des paires (image, label).

    Returns:
        None: Affiche une figure matplotlib avec une image par classe.
    """
    labels_seen = set()
    images_to_show = []
    for image, label in dataset:
        if label not in labels_seen:
            images_to_show.append((image, label))
            labels_seen.add(label)
        if len(labels_seen) == 10:
            break
    plt.figure(figsize=(10, 5))
    for i, (image, label) in enumerate(images_to_show):
        plt.subplot(2, 5, i + 1)
        plt.imshow(image.squeeze(), cmap="gray")
        plt.title(f"Label: {label}")
        plt.axis("off")
    plt.tight_layout()
    plt.show()


def plot_data_distribution(
    dataset: List[Tuple[np.ndarray, int]], title: str = "Répartition des données"
) -> None:
    """Affiche un histogramme représentant la répartition des classes
    dans un dataset. Permet de détecter rapidement d’éventuels déséquilibres
    entre les classes.

    Args:
        dataset (List[Tuple[np.ndarray, int]]): Liste de tuples (image, label),
        les images étant typiquement des numpy arrays.
        title (str, optional): Titre du graphique. Par défaut :
        "Répartition des données".

    Returns:
        None: Affiche un graphique matplotlib à l’écran.
    """
    labels = [label for _, label in dataset]
    label_counts = Counter(labels)
    plt.figure(figsize=(10, 5))
    plt.bar(label_counts.keys(), label_counts.values(), color="skyblue")
    plt.title(title)
    plt.xlabel("Classe")
    plt.ylabel("Nombre d'exemples")
    plt.xticks(range(10))
    plt.show()
