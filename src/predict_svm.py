import pickle
from src.preprocess import clean_text



with open("models/svm_model.pkl","rb") as f:
    model = pickle.load(f)

with open("models/svm_vectorizer.pkl","rb") as f:
    vectorizer = pickle.load(f)

def predict_hate_svm(text):

    cleaned_text = clean_text(text)

    vectorized_text = vectorizer.transform([cleaned_text])

    prediction = model.predict(vectorized_text)[0]

    probs = model.predict_proba(vectorized_text)[0]

    labels = {
        0: "Hate Speech",
        1: "Offensive Language",
        2: "Neither"
    }
    confidence = max(probs) * 100

    return labels[prediction], round(confidence,2)

    