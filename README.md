# Projet Deep Learning - Classification MNIST avec PyTorch & VAT

Ce projet implémente un modèle de classification d’images basé sur le dataset **MNIST**, en combinant **apprentissage supervisé** et **apprentissage semi-supervisé** avec la technique de **Virtual Adversarial Training (VAT)**.  
Le projet a été structuré de manière modulaire et est équipé de tests, CI/CD et d’un environnement Dockerisé.

---

## Objectifs

- Entraîner un modèle robuste avec **peu de données étiquetées** (100 images, soit 10 par classe).
- Exploiter les données non-étiquetées via **VAT**, une méthode d’**apprentissage semi-supervisé**.
- Obtenir une bonne généralisation avec peu de supervision explicite.
- Intégrer une pipeline de test et de qualité de code.

---

## Contenu du projet

Le projet comprend :

- **`src/`** : Code source principal avec les modules pour la gestion des données, la définition des modèles, l’entraînement avec VAT, et l’implémentation de la Virtual Adversarial Training.
- **`tests/`** : Tests unitaires réalisés avec `pytest` pour valider chaque composant.
- **`notebooks/** : Notebooks Jupyter illustrant le pipeline complet, de la préparation des données à l’évaluation.
- **`best_model.pth`** : Poids sauvegardés du meilleur modèle entraîné.
- **`.github/workflows/ci.yml`** : Configuration GitHub Actions pour l’intégration continue (tests et linting).
- **`Dockerfile`** : Environnement Docker assurant la portabilité et la reproductibilité.
- **`requirements.txt`** : Liste des dépendances Python nécessaires.
- **`README.md`** : Ce document décrivant le projet et son utilisation.

---

## 🧪 Méthodologie

### 🔸 Données
- **MNIST** : 28×28 pixels, chiffres manuscrits.
- 100 échantillons étiquetés (10 par classe).
- Reste utilisé comme données non-étiquetées.
- Jeu de test standard.

### 🔸 Modèles
- `SimpleCNNWithDropout` : un petit CNN avec Dropout.
- `PretrainedResNetForMNIST` : adaptation de ResNet18 (ImageNet → MNIST).
- `PretrainedMobileNetForMNIST` : adaptation de MobileNetV2.

### 🔸 Apprentissage semi-supervisé avec VAT
- La **Virtual Adversarial Loss** est utilisée pour guider l'entraînement sur les données non étiquetées.
- La fonction `virtual_adversarial_loss` est définie dans `src/vat.py`.

---

## ⚙️ Environnement & Linting

Le projet utilise :

- **Python 3.11**
- Linting avec :
  - [`flake8`](https://flake8.pycqa.org/)
  - [`black`](https://black.readthedocs.io/)
  - [`isort`](https://pycqa.github.io/isort/)
  - [`mypy`](https://mypy.readthedocs.io/)
- Tests unitaires avec `pytest`

### 🧪 Exécuter les tests :
pytest --disable-warnings

---
### Verifier le code:
flake8 src/ tests/
black --check .
isort --check-only .
mypy src/

### Intégration Continue (CI)
Une action GitHub automatique (.github/workflows/ci.yml) :
installe les dépendances
vérifie la qualité du code
exécute les tests unitaires
Elle se déclenche sur chaque push ou pull request vers main.

### Docker
docker build -t mnist-vat .
docker run -it mnist-vat

### Installation locale
git clone https://github.com/ton-repo/deep-learning-vat.git
cd deep-learning-vat
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

## ⚙️ Technologies utilisées

- [PyTorch](https://pytorch.org/)
- Torch
- NumPy
- matplotlib
- scikit-learn

## Auteurs

- **Aymane Mimoun**
- **Mohtadi Hammami**
- **Alexandre Combeau**

Master Data Science - Université Paris-Saclay 2025
