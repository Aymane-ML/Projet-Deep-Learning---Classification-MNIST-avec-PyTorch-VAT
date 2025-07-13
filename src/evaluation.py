import torch
import matplotlib.pyplot as plt
import numpy as np
from torch import nn
from torch.utils.data import DataLoader
from typing import Tuple
import numpy.typing as npt
from sklearn.metrics import confusion_matrix, classification_report

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate(model: nn.Module, test_loader: DataLoader) -> None:
    """
    Évalue la précision brute du modèle sur un jeu de test.

    Args:
        model (nn.Module): Modèle PyTorch à évaluer.
        test_loader (DataLoader): Contient les échantillons de test (x, y).

    Returns:
        None: Affiche la précision en pourcentage.
    """
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            _, predicted = torch.max(outputs.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
    accuracy = 100 * correct / total
    print(f"Précision sur les données de test : {accuracy:.2f}%")


def evaluate_metrics(
    model: nn.Module, test_loader: DataLoader
) -> Tuple[npt.NDArray, str]:
    """Calcule la matrice de confusion et le rapport de classification
    (F1, précision, rappel).

    Args:
        model (nn.Module): Modèle à évaluer.
        test_loader (DataLoader): Données de test (x, y).

    Returns:
        Tuple[npt.NDArray, str]:
            - Matrice de confusion (np.ndarray)
            - Rapport de classification (str)
    """
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    conf_matrix = confusion_matrix(all_labels, all_preds)
    class_report = classification_report(all_labels, all_preds, digits=4)
    print("Matrice de Confusion :\n", conf_matrix)
    print("\nRapport de Classification :\n", class_report)
    return conf_matrix, class_report


def visualize_errors(
    model: nn.Module, test_loader: DataLoader, num_errors: int = 10
) -> None:
    """
    Affiche des exemples d’erreurs de classification du modèle.

    Args:
        model (nn.Module): Modèle entraîné à évaluer.
        test_loader (DataLoader): Données de test.
        num_errors (int): Nombre d’erreurs à afficher.Defaults to 10.

    Returns:
        None: Affiche une figure matplotlib contenant les erreurs.
    """
    model.eval()
    errors = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            for i in range(len(labels)):
                if predicted[i] != labels[i]:
                    errors.append(
                        (inputs[i].cpu(), predicted[i].cpu(), labels[i].cpu())
                    )
    if not errors:
        print("Aucune erreur de classification détectée.")
        return
    plt.figure(figsize=(10, 5))
    for i, (image, pred, true) in enumerate(errors[:num_errors]):
        plt.subplot(2, 5, i + 1)
        plt.imshow(image.squeeze(), cmap="gray")
        plt.title(f"Préd : {pred}, Vrai : {true}")
        plt.axis("off")
    plt.tight_layout()
    plt.show()
