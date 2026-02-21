from fastapi import FastAPI
from pydantic import BaseModel
import joblib

app = FastAPI()

# Load model and vectorizer
model = joblib.load("fake_news_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

class NewsInput(BaseModel):
    text: str

@app.post("/predict")
def predict(news: NewsInput):
    vectorized_text = vectorizer.transform([news.text])
    prediction = model.predict(vectorized_text)[0]
    probability = model.predict_proba(vectorized_text)[0][1]

    return {
        "prediction": int(prediction),
        "fake_probability": float(probability)
    }