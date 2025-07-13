import torch.nn as nn
from torchvision import models
from torch import Tensor


class SimpleCNNWithDropout(nn.Module):
    """
    Un CNN simple avec deux couches convolutionnelles et Dropout.
    Adapté à MNIST.
    """

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(64 * 5 * 5, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x: Tensor) -> Tensor:
        """Passe avant du modèle.

        Args:
            x (Tensor): Entrée d'image de forme (B, 1, 28, 28)

        Returns:
            Tensor: Logits de sortie (B, 10)
        """
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = nn.functional.relu(self.conv2(x))
        x = nn.functional.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = x.view(-1, 64 * 5 * 5)
        x = nn.functional.relu(self.fc1(x))
        x = self.dropout2(x)
        return self.fc2(x)


class PretrainedMobileNetForMNIST(nn.Module):
    """Adaptation de MobileNetV2 préentraîné pour la classification MNIST
    (images 1 canal, 10 classes).

    Modifications :
    - Première couche modifiée pour accepter des images en niveaux de gris
    (1 canal).
    - Dernière couche remplacée par un classifieur adapté à MNIST.
    """

    def __init__(self) -> None:
        super(PretrainedMobileNetForMNIST, self).__init__()
        self.mobilenet = models.mobilenet_v2(pretrained=True)
        self.mobilenet.features[0][0] = nn.Conv2d(
            1, 32, kernel_size=3, stride=2, padding=1, bias=False
        )
        num_ftrs = self.mobilenet.last_channel
        self.mobilenet.classifier = nn.Sequential(
            nn.Dropout(0.2), nn.Linear(num_ftrs, 10)
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Propagation avant du modèle.

        Args:
            x (Tensor):  Image d'entrée de taille (batch_size, 1, 28, 28).

        Returns:
            Tensor: Logits de sortie (batch_size, 10).
        """
        return self.mobilenet(x)
