# Spotify Hit & Mood Engine

Predicts whether a track is a "hit" (popularity >= 65) and assigns it a mood cluster.
Models: Logistic Regression, KNN, tuned Gradient Boosting. PCA (2D map) + K-Means (moods). Streamlit UI.

## Run locally
1. Download `dataset.csv` from https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset and put it in this folder.
2. `pip install -r requirements.txt`
3. `python train.py`   (creates `models/bundle.joblib`, `models/results.csv`, plots)
4. `streamlit run app.py`

## Deployed using Streamlit
https://spotify-hit.streamlit.app/
