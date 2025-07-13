# Étape 1 : Image de base avec PyTorch + torchvision
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

# Étape 2 : Définir le dossier de travail
WORKDIR /app

# Étape 3 : Copier les fichiers
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copier le code source
COPY src/ ./src
COPY tests/ ./tests

# Étape 4 : Exécuter les tests automatiquement (optionnel)
CMD ["pytest", "--maxfail=1", "--disable-warnings", "-q"]
