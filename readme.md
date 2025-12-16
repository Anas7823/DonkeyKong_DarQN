# 🦍 Donkey Kong AI - Dueling DARQN

Ce projet implémente un agent d'Apprentissage par Renforcement (Reinforcement Learning) capable d'apprendre à jouer au jeu Atari **Donkey Kong** à partir de zéro.

Le modèle utilise une architecture avancée combinant plusieurs techniques de l'état de l'art pour gérer la complexité visuelle et temporelle du jeu.

## 🧠 Architecture du Modèle

L'agent n'utilise pas un simple DQN standard, mais une architecture **Dueling DARQN** (Deep Attention Recurrent Q-Network) :

1.  **CNN (Convolutional Neural Network)** : Extrait les caractéristiques visuelles de chaque frame (image du jeu).
2.  **LSTM (Long Short-Term Memory)** : Traite une séquence de frames pour comprendre le mouvement et la temporalité (vitesse des barils, direction de Mario).
3.  **Multi-Head Attention** : Permet au modèle de se "concentrer" sur les zones importantes de l'écran (ex: Mario vs les Barils) à différents moments.
4.  **Dueling Network** : Sépare l'estimation de la valeur de l'état $V(s)$ et l'avantage de l'action $A(s, a)$ pour une convergence plus stable.
5.  **Double DQN** : Réduit la surestimation des Q-values.

## 🛠️ Prérequis et Installation

Le projet nécessite **Python 3.8+**.

### 1. Cloner ou télécharger le projet
Placez les fichiers `train.DonkeyKong_DarQN.ipynb` et `play.py` dans un dossier.

### 2. Installer les dépendances
Installez les bibliothèques nécessaires, y compris Gymnasium et l'émulateur Atari (ALE) :

```bash
pip install gymnasium[atari] ale-py shimmy opencv-python tensorflow numpy