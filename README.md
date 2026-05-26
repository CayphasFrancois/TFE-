## 1. Créer un environnement virtuel

```bash
python -m venv nom
```

## 2. Activer l’environnement virtuel

```bash
.\nom\Scripts\activate
```

## 3. Installer les dépendances

```bash
pip install -r requirements.txt
```

## 4. Lancer le code Environnement_setup.py pour créer les environnements d'entraînement, validation et test :

```bash
python Environnement_setup.py
```
## 5. Lancer un des 3 codes TD3_Xactions

Pour les codes TD3_1action_PQ.py et TD3_1action_sans_PQ.py, il est possible de choisir la fonction de coût en modifiant la variable "reward_function" (V_MSE ou V_complexe).
