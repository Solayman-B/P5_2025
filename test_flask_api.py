import os
import sys
import tempfile
import pickle
import pytest
import numpy as np
from flask import json
import flask_api

# Patch sys.path so we can import the app
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

@pytest.fixture
def client(monkeypatch):
    # Patch vectorizer and model with mocks
    class DummyVectorizer:
        def transform(self, texts):
            # Return a dummy numpy array
            return np.array([[1, 2, 3]])

    class DummyModel:
        def __init__(self):
            self.classes_ = np.array(['a', 'b', 'c'])
        def predict(self, X):
            return [[1, 0, 1]]
        def predict_proba(self, X):
            # Return probabilities for 3 classes
            return [np.array([0.1, 0.7, 0.2])]

    monkeypatch.setattr(flask_api, "vectorizer", DummyVectorizer())
    monkeypatch.setattr(flask_api, "model", DummyModel())
    monkeypatch.setattr(flask_api, "label_names", ["tag1", "tag2", "tag3"])

    with flask_api.app.test_client() as client:
        yield client

def test_health_ok(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["ok"] is True
    assert data["vectorizer_loaded"] is True
    assert data["model_loaded"] is True

def test_predict_success(client):
    resp = client.post("/predict", json={"text": "This is a test sentence."})
    assert resp.status_code == 200
    data = resp.get_json()
    assert "predicted" in data
    assert isinstance(data["predicted"], list)
    assert "probabilities" in data
    assert isinstance(data["probabilities"], dict)
    assert "top_tags" in data
    assert isinstance(data["top_tags"], list)
    assert set(data["probabilities"].keys()) == {"tag1", "tag2", "tag3"}
    assert data["top_tags"][0] == "tag2"  # highest probability

def test_predict_missing_text(client):
    resp = client.post("/predict", json={})
    assert resp.status_code == 400
    data = resp.get_json()
    assert "error" in data

def test_predict_no_json(client):
    resp = client.post("/predict", data="notjson", content_type="text/plain")
    assert resp.status_code == 400
    data = resp.get_json()
    assert "error" in data

def test_predict_model_not_loaded(monkeypatch):
    # Patch vectorizer/model to None
    monkeypatch.setattr(flask_api, "vectorizer", None)
    monkeypatch.setattr(flask_api, "model", None)
    with flask_api.app.test_client() as client:
        resp = client.post("/predict", json={"text": "test"})
        assert resp.status_code == 500
        data = resp.get_json()
        assert "error" in data

def test_to_native_numpy_types():
    arr = np.array([1, 2, 3])
    d = {"a": np.int32(1), "b": arr}
    result = flask_api.to_native(d)
    assert result == {"a": 1, "b": [1, 2, 3]}