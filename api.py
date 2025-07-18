from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import pickle
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
import numpy as np
from typing import List, Dict
import os

app = FastAPI(title="Stack Overflow Tag Predictor", version="1.0.0")

# Variables globales pour le modèle et le vectorizer
model = None
vectorizer = None
label_columns = None

class QuestionRequest(BaseModel):
    question: str

class PredictionResponse(BaseModel):
    question: str
    predicted_tags: List[str]
    confidence_scores: Dict[str, float]

def clean_text(text: str) -> str:
    """
    Fonction de préprocessing du texte
    """
    if pd.isna(text) or text == "":
        return ""
    
    # Convertir en minuscules
    text = text.lower()
    
    # Supprimer les balises HTML
    text = re.sub(r'<[^>]+>', ' ', text)
    
    # Supprimer les caractères spéciaux mais garder les espaces
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    
    # Supprimer les espaces multiples
    text = re.sub(r'\s+', ' ', text)
    
    # Supprimer les espaces en début et fin
    text = text.strip()
    
    return text

def load_model_and_data():
    """
    Charger le modèle et les données depuis les fichiers pickle
    """
    global model, vectorizer, label_columns
    
    try:
        # Vérifier si tous les fichiers existent
        if not all(os.path.exists(f) for f in ["model.pkl", "vectorizer.pkl", "labels.pkl"]):
            raise FileNotFoundError("Fichiers de modèle manquants (model.pkl, vectorizer.pkl, labels.pkl)")
        
        # Charger le modèle pré-entraîné
        with open("model.pkl", "rb") as f:
            model = pickle.load(f)
        with open("vectorizer.pkl", "rb") as f:
            vectorizer = pickle.load(f)
        with open("labels.pkl", "rb") as f:
            label_columns = pickle.load(f)
            
        print(f"Modèle chargé avec succès - {len(label_columns)} tags disponibles")
        print(f"Tags: {label_columns}")
        
    except Exception as e:
        print(f"Erreur lors du chargement du modèle: {e}")
        raise HTTPException(status_code=500, detail=f"Impossible de charger le modèle: {str(e)}")

@app.on_event("startup")
async def startup_event():
    """
    Charger le modèle au démarrage de l'API
    """
    load_model_and_data()

@app.get("/")
async def root():
    """
    Endpoint racine
    """
    return {
        "message": "Stack Overflow Tag Predictor API", 
        "version": "1.0.0",
        "endpoints": {
            "/predict": "POST - Prédire les tags pour une question",
            "/health": "GET - Vérifier l'état de l'API",
            "/tags": "GET - Liste des tags disponibles"
        }
    }

@app.get("/health")
async def health_check():
    """
    Vérifier l'état de l'API
    """
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "vectorizer_loaded": vectorizer is not None,
        "available_tags": len(label_columns) if label_columns else 0
    }

@app.get("/tags")
async def get_available_tags():
    """
    Récupérer la liste des tags disponibles
    """
    if label_columns is None:
        raise HTTPException(status_code=500, detail="Modèle non chargé")
    
    return {
        "available_tags": label_columns,
        "count": len(label_columns)
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict_tags(request: QuestionRequest):
    """
    Prédire les tags pour une question
    """
    if model is None or vectorizer is None or label_columns is None:
        raise HTTPException(status_code=500, detail="Modèle non chargé")
    
    try:
        # Préprocesser la question
        cleaned_question = clean_text(request.question)
        
        if not cleaned_question.strip():
            raise HTTPException(status_code=400, detail="Question vide après préprocessing")
        
        # Transformer en features TF-IDF
        question_tfidf = vectorizer.transform([cleaned_question])
        
        # Prédire les probabilités
        probabilities = model.predict_proba(question_tfidf)
        
        # Récupérer les prédictions binaires
        predictions = model.predict(question_tfidf)[0]
        
        # Créer le dictionnaire des scores de confiance
        confidence_scores = {}
        predicted_tags = []
        
        for i, (label, pred) in enumerate(zip(label_columns, predictions)):
            # Récupérer la probabilité de la classe positive (1)
            if hasattr(probabilities[i], 'shape') and len(probabilities[i][0]) > 1:
                confidence = float(probabilities[i][0][1])  # Probabilité classe 1
            else:
                confidence = float(pred)  # Fallback
            
            confidence_scores[label] = confidence
            
            # Ajouter aux tags prédits si la prédiction est positive
            if pred == 1:
                predicted_tags.append(label)
        
        # Si aucun tag prédit, prendre les 3 meilleurs scores
        if not predicted_tags:
            sorted_tags = sorted(confidence_scores.items(), key=lambda x: x[1], reverse=True)
            predicted_tags = [tag for tag, _ in sorted_tags[:3]]
        
        return PredictionResponse(
            question=request.question,
            predicted_tags=predicted_tags,
            confidence_scores=confidence_scores
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de la prédiction: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)