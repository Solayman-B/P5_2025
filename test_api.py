"""
Script de test pour l'API de prédiction de tags
"""
import requests
import json

# URL de base de l'API
BASE_URL = "http://localhost:8000"

def test_api():
    """
    Tester les différents endpoints de l'API
    """
    
    # 1. Test de l'endpoint racine
    print("1. Test de l'endpoint racine...")
    try:
        response = requests.get(f"{BASE_URL}/")
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
        print()
    except Exception as e:
        print(f"Erreur: {e}")
        return
    
    # 2. Test du health check
    print("2. Test du health check...")
    try:
        response = requests.get(f"{BASE_URL}/health")
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
        print()
    except Exception as e:
        print(f"Erreur: {e}")
    
    # 3. Test de la liste des tags
    print("3. Test de la liste des tags...")
    try:
        response = requests.get(f"{BASE_URL}/tags")
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
        print()
    except Exception as e:
        print(f"Erreur: {e}")
    
    # 4. Test de prédiction
    print("4. Test de prédiction...")
    test_questions = [
        "How to create a web scraper in Python?",
        "What is the difference between let and var in JavaScript?",
        "How to style a div with CSS?",
        "SQL query to join two tables",
        "React component lifecycle methods"
    ]
    
    for question in test_questions:
        try:
            payload = {"question": question}
            response = requests.post(f"{BASE_URL}/predict", json=payload)
            print(f"Question: {question}")
            print(f"Status: {response.status_code}")
            if response.status_code == 200:
                result = response.json()
                print(f"Tags prédits: {result['predicted_tags']}")
                print(f"Scores de confiance: {result['confidence_scores']}")
            else:
                print(f"Erreur: {response.text}")
            print("-" * 50)
        except Exception as e:
            print(f"Erreur lors de la prédiction: {e}")

if __name__ == "__main__":
    print("Démarrage des tests de l'API...")
    print("Assurez-vous que l'API est démarrée avec: python api.py")
    print("=" * 60)
    test_api()
