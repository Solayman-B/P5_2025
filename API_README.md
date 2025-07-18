# API de Prédiction de Tags Stack Overflow

Cette API FastAPI permet de prédire les tags pour des questions Stack Overflow en utilisant TF-IDF et la régression logistique.

## Installation

1. Installer les dépendances :
```bash
pip install -r requirements.txt
```

## Utilisation

### 1. Entraîner le modèle (optionnel)

```bash
python train_model.py
```

### 2. Démarrer l'API

```bash
python start_api.py
```

Ou directement :
```bash
python api.py
```

L'API sera disponible sur `http://localhost:8000`

### 3. Tester l'API

```bash
python test_api.py
```

## Endpoints

### GET `/`
Point d'entrée principal avec les informations de l'API.

### GET `/health`
Vérification de l'état de l'API et du modèle.

### GET `/tags`
Liste des tags disponibles pour la prédiction.

### POST `/predict`
Prédire les tags pour une question.

**Payload :**
```json
{
    "question": "How to create a web scraper in Python?"
}
```

**Réponse :**
```json
{
    "question": "How to create a web scraper in Python?",
    "predicted_tags": ["python", "web-scraping"],
    "confidence_scores": {
        "python": 0.85,
        "javascript": 0.12,
        "web-scraping": 0.78,
        ...
    }
}
```

## Documentation Interactive

Une fois l'API démarrée, vous pouvez accéder à :
- Documentation Swagger : `http://localhost:8000/docs`
- Documentation ReDoc : `http://localhost:8000/redoc`

## Fonctionnalités

### Préprocessing
- Suppression des balises HTML
- Conversion en minuscules
- Suppression des caractères spéciaux
- Normalisation des espaces

### Modèle
- **Vectorisation** : TF-IDF avec n-grammes (1,2)
- **Classification** : Régression logistique multi-output
- **Sauvegarde** : Modèle persistant en fichiers pickle

### API
- **Framework** : FastAPI avec validation Pydantic
- **Réponses** : JSON avec tags prédits et scores de confiance
- **Erreurs** : Gestion robuste des erreurs
- **Documentation** : Auto-générée avec Swagger/OpenAPI

## Structure des fichiers

```
├── api.py              # API FastAPI principale
├── start_api.py        # Script de démarrage
├── train_model.py      # Entraînement du modèle
├── test_api.py         # Tests de l'API
├── requirements.txt    # Dépendances Python
├── model.pkl          # Modèle entraîné (généré)
├── vectorizer.pkl     # Vectorizer TF-IDF (généré)
├── labels.pkl         # Liste des labels (généré)
└── cleaned_data.xlsx  # Données d'entraînement
```

## Exemples d'utilisation

### Avec curl
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"question": "How to use pandas dataframe?"}'
```

### Avec Python requests
```python
import requests

response = requests.post(
    "http://localhost:8000/predict",
    json={"question": "How to use pandas dataframe?"}
)
print(response.json())
```

## Notes

- Si le fichier `cleaned_data.xlsx` n'est pas trouvé, un modèle factice sera créé pour les tests
- Le modèle est automatiquement chargé au démarrage de l'API
- Les scores de confiance indiquent la probabilité d'appartenance à chaque tag
