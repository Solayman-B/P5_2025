import streamlit as st
import requests
import pandas as pd

st.set_page_config(page_title="P5 Tag Predictor", layout="centered")

st.title("Interface de test - API de prédiction de tags")
st.write("Entrez une question (ou un extrait) et appelez l'endpoint /predict de l'API Flask.")

api_url = st.text_input("URL de l'API", value="http://localhost:5000/predict")

with st.form("predict_form"):
    text = st.text_area("Question", height=150, placeholder="Écrivez votre question ici...")
    submitted = st.form_submit_button("Prédire")

if submitted:
    if not text.strip():
        st.error("La question est vide — merci d'entrer du texte.")
    else:
        payload = {"text": text}
        try:
            with st.spinner("Appel de l'API..."):
                resp = requests.post(api_url, json=payload, timeout=10)
            if resp.status_code != 200:
                st.error(f"Erreur API: {resp.status_code} - {resp.text}")
            else:
                data = resp.json()
                st.success("Réponse reçue")

                # Afficher prédictions
                st.subheader("Prédiction brute")
                st.json(data)

                # Afficher predicted
                if "predicted" in data:
                    st.subheader("Predicted")
                    st.write(data["predicted"])

                # Afficher top_tags
                if "top_tags" in data and data["top_tags"]:
                    st.subheader("Top tags")
                    st.write(data["top_tags"])

                # Afficher probabilités sous forme de tableau trié
                if "probabilities" in data and isinstance(data["probabilities"], dict) and data["probabilities"]:
                    probs = data["probabilities"]
                    df = pd.DataFrame(list(probs.items()), columns=["tag", "probability"])
                    df = df.sort_values("probability", ascending=False).reset_index(drop=True)
                    st.subheader("Probabilités par tag")
                    st.table(df)
        except requests.exceptions.RequestException as e:
            st.error(f"Échec de l'appel à l'API: {e}")

st.markdown("---")
st.write("Remarques :")
st.write("- Démarrer d'abord l'API Flask (ex: python flask_api.py)")
st.write("- Si l'API est sur un autre hôte/port, modifiez l'URL en haut.")
