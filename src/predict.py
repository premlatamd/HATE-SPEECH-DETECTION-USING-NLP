import pickle
from src.preprocess import clean_text

with open("models/model.pkl","rb") as f:
    model=pickle.load(f)

with open("models/vectorizer.pkl","rb") as f:
    vectorizer=pickle.load(f)

def predict_hate(text):
    cleaned_text = clean_text(text)
    vectorized_text = vectorizer.transform([cleaned_text])

    prediction = model.predict(vectorized_text)[0]
    #print("Raw prediction:", prediction)
    probs = model.predict_proba(vectorized_text)[0]

    labels = {
        0: "Hate Speech",
        1: "Offensive Language",
        2: "Neither"
    }
    confidence = max(probs) * 100

    #return labels[prediction], round(confidence,2)

    return labels[prediction], probs


if __name__ == "__main__":

    text = "I hate all immigrants"

    result, probs = predict_hate(text)

    print("Probabilities:", probs)
    print("Prediction:", result)