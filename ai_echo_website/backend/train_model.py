# backend/train_model.py

import pandas as pd
from pathlib import Path
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multioutput       import MultiOutputRegressor
from sklearn.ensemble          import RandomForestRegressor

# ───────────────────────── CONFIG ─────────────────────────
BASE_DIR    = Path(__file__).parent
EXCEL_FILE  = BASE_DIR / "Words.xlsx"

VEC_PICKLE  = BASE_DIR / "tfidf_vectorizer.pkl"
MODEL_PICKLE= BASE_DIR / "text_prediction_model.pkl"

# ───────────────────────── LOAD DATA ────────────────────────
# assumes first column is the phrase, next three columns are Stability, Similarity, Spead
df = pd.read_excel(EXCEL_FILE, engine="openpyxl")
phrases = df.iloc[:,0].astype(str).tolist()
y       = df[["Stability", "Similarity", "Spead"]].values

# ───────────────────────── VECTORIZE ───────────────────────
print("⏳  Fitting TF-IDF vectorizer…")
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(phrases)

# ───────────────────────── TRAIN MODEL ──────────────────────
print("⏳  Training RandomForest multi-output regressor…")
model = MultiOutputRegressor(
    RandomForestRegressor(n_estimators=300, random_state=42)
)
model.fit(X, y)

# ───────────────────────── DUMP PICKLES ────────────────────
print(f"💾  Saving vectorizer to {VEC_PICKLE.name}")
joblib.dump(vectorizer, VEC_PICKLE)

print(f"💾  Saving model to {MODEL_PICKLE.name}")
joblib.dump(model, MODEL_PICKLE)

print("✅  Training complete. Pickles are ready for scikit‑learn 1.6.1.")
