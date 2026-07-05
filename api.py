

from fastapi import FastAPI
from src.predict import predict_hate
from src.rf_predict import rf_predict_hate
from src.predict_svm import predict_hate_svm

app = FastAPI()

@app.get("/")
def home():
    return {"message": "Hate Speech API Running"}

@app.get("/predict")
def predict(text: str, model_type: str):

    if model_type == "LOGISTIC REGRESSION":
        result = predict_hate(text)

    elif model_type == "RANDOM FOREST":
        result = rf_predict_hate(text)
    
    elif model_type == "SVM":
        result = predict_hate_svm(text)
        
    else:
        return {"error": "invalid Model"}

    return {
        "prediction": result[0],"confidence":result[1]
    }