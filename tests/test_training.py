import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import SGD
from src.training import train
from src.models import SimpleCNNWithDropout


def get_dummy_loaders(
    batch_size=8
):
    """Crée des loaders factices pour tests avec images MNIST
    (1 canal, 28x28).
    """
    x_labeled = torch.randn(batch_size * 2, 1, 28, 28)
    y_labeled = torch.randint(0, 10, (batch_size*2,))
    x_unlabeled = torch.randn(batch_size*2, 1, 28, 28)
    labeled_ds = TensorDataset(x_labeled, y_labeled)
    unlabeled_ds = TensorDataset(
        x_unlabeled,
        torch.zeros_like(y_labeled)
    )
    labeled_loader = DataLoader(labeled_ds, batch_size=batch_size)
    unlabeled_loader = DataLoader(unlabeled_ds, batch_size=batch_size)
    return labeled_loader, unlabeled_loader


def test_train_runs_without_error():
    """Vérifie que la fonction train s'exécute sans lever d'erreur.
    """
    model = SimpleCNNWithDropout()
    labeled_loader, unlabeled_loader = get_dummy_loaders()
    optimizer = SGD(model.parameters(), lr=0.01)
    try:
        train(
            model,
            labeled_loader,
            unlabeled_loader,
            optimizer,
            alpha=0.1,
            epochs=1,
            device='cpu'
        )
    except Exception as e:
        raise AssertionError(f"Erreur dans train(): {e}")


def test_train_computes_gradients():
    """Vérifie que les gradients sont calculés après train().
    """
    model = SimpleCNNWithDropout()
    labeled_loader, unlabeled_loader = get_dummy_loaders()
    optimizer = SGD(model.parameters(), lr=0.01)
    train(
        model,
        labeled_loader,
        unlabeled_loader,
        optimizer,
        alpha=0.1,
        epochs=1,
        device='cpu'
    )
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert (
        any(g.abs().sum().item() > 0 for g in grads)
    ), "Aucun gradient calculé"
