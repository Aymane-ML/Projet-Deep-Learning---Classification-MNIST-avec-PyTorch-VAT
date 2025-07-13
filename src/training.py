import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from src.vat import virtual_adversarial_loss


def train(
    model: nn.Module,
    labeled_loader: DataLoader,
    unlabeled_loader: DataLoader,
    optimizer: Optimizer,
    alpha: float = 1.0,
    epochs: int = 10,
    device: str = "cpu",
) -> None:
    """Entraîne un modèle avec apprentissage semi-supervisé
    (Virtual Adversarial Training).

    Args:
        model (nn.Module): Modèle à entraîner.
        labeled_loader (DataLoader): Données étiquetées (x, y).
        unlabeled_loader (DataLoader): Données non étiquetées (x, _).
        optimizer (Optimizer): Optimiseur utilisé.
        alpha (float): Poids de la perte non supervisée (VAT).
        epochs (int): Nombre d’époques d'entraînement.
        device (str): Appareil cible ('cpu' ou 'cuda').
    """
    model.to(device)
    model.train()
    criterion = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        total_loss = 0
        for (x_labeled, y_labeled), (x_unlabeled, _) in zip(
            labeled_loader, unlabeled_loader
        ):
            x_labeled, y_labeled = x_labeled.to(device), y_labeled.to(device)
            x_unlabeled = x_unlabeled.to(device)
            y_pred = model(x_labeled)
            supervised_loss = criterion(y_pred, y_labeled)
            vat_loss = virtual_adversarial_loss(model, x_unlabeled)
            loss = supervised_loss + alpha * vat_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Époque {epoch+1}/{epochs}, Perte totale: {total_loss:.4f}")
