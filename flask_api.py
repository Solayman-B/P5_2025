from flask import Flask, request, jsonify
import pickle
import re
import os
import numpy as np

app = Flask(__name__)



import spacy, re
# Load spaCy's English model.
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")
version_pattern = re.compile(r'^\d+(\.\d+)+$')
def clean_text(text):
    doc = nlp(text.lower())
    tokens = [
        token.lemma_
        for token in doc
        if (
            (token.is_alpha or token.text.isalnum())
            and not token.is_stop
            and not token.pos_ == "PRON"
            and not version_pattern.match(token.text)
        )
    ]
    return " ".join(tokens)


# remove_html_tags: simple regex-based fallback (avoid BeautifulSoup dependency)
def remove_html_tags(text):
    if not text:
        return ""
    text = re.sub(r'<[^>]+>', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()

with open("notebooks/countvectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

with open("notebooks/logistic_regression_model.pkl", "rb") as f:
    model = pickle.load(f)

# Charger les noms de labels si présents (labels.pkl)
label_names = None
for lp in ("notebooks/labels.pkl", "labels.pkl", "notebooks/label_names.pkl"):
    if os.path.exists(lp):
        with open(lp, 'rb') as lf:
            label_names = pickle.load(lf)
        break

# Si labels non fournis et modèle multi-output, créer des noms génériques
if label_names is None and model is not None and hasattr(model, 'estimators_'):
    try:
        n = len(model.estimators_)
        label_names = [f"tag_{i}" for i in range(n)]
    except Exception:
        label_names = None

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'ok': True,
        'vectorizer_loaded': vectorizer is not None,
        'model_loaded': model is not None
    })

@app.route('/predict', methods=['POST'])
def predict():
    if vectorizer is None or model is None:
        return jsonify({'error': 'vectorizer or model not loaded'}), 500
    data = request.get_json(silent=True)
    if not data or 'text' not in data:
        return jsonify({'error': "missing 'text' in body"}), 400
    # preprocess text
    text = clean_text(data['text'])
    X = vectorizer.transform([text])
    # prediction (binary vector per label)
    pred = model.predict(X)
    result = {'predicted': None}
    try:
        if isinstance(pred, (list, tuple)):
            result['predicted'] = pred[0]
        else:
            result['predicted'] = pred[0] if len(pred) else None
    except Exception:
        result['predicted'] = str(pred)

    # Build probability map tag -> prob_positive
    prob_map = {}
    try:
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(X)
            # Case: list (multioutput) or list with single array
            if isinstance(proba, list):
                # If it's a multioutput list where each element corresponds to a label
                if label_names and len(proba) == len(label_names):
                    for i, p in enumerate(proba):
                        arr = np.asarray(p)
                        arr0 = arr[0] if arr.ndim == 2 else arr
                        prob_pos = float(arr0[1]) if arr0.size > 1 else float(arr0[0])
                        tag = label_names[i]
                        prob_map[tag] = prob_pos
                else:
                    # Could be a single estimator returning [array_of_probs]
                    arr = np.asarray(proba[0])
                    arr0 = arr[0] if arr.ndim == 2 else arr
                    if label_names and len(label_names) == arr0.size:
                        for i, tag in enumerate(label_names):
                            prob_map[tag] = float(arr0[i])
                    else:
                        classes = getattr(model, 'classes_', None)
                        if classes is not None and len(classes) == arr0.size:
                            for i, c in enumerate(classes):
                                prob_map[str(c)] = float(arr0[i])
                        else:
                            for i, val in enumerate(arr0):
                                prob_map[f'class_{i}'] = float(val)
            else:
                # proba is ndarray
                arr = np.asarray(proba)
                arr0 = arr[0] if arr.ndim == 2 else arr
                if label_names and len(label_names) == arr0.size:
                    for i, tag in enumerate(label_names):
                        prob_map[tag] = float(arr0[i])
                else:
                    classes = getattr(model, 'classes_', None)
                    if classes is not None and len(classes) == arr0.size:
                        for i, c in enumerate(classes):
                            prob_map[str(c)] = float(arr0[i])
                    else:
                        for i, val in enumerate(arr0):
                            prob_map[f'class_{i}'] = float(val)
    except Exception:
        pass

    # Select top-3 tags with probability
    top_tags = []
    if prob_map:
        sorted_tags = sorted(prob_map.items(), key=lambda x: x[1], reverse=True)
        top_tags = [tag for tag, _ in sorted_tags[:3]]

    result['probabilities'] = prob_map
    result['top_tags'] = top_tags

    # Convert all numpy types to Python types
    result = to_native(result)

    return jsonify(result)

def to_native(o):
    """Recursively convert numpy types/arrays to Python natives for JSON serialization."""
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, dict):
        return {k: to_native(v) for k, v in o.items()}
    if isinstance(o, list):
        return [to_native(v) for v in o]
    return o

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
