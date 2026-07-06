import streamlit as st
import requests
from src.predict import predict_hate
from src.evaluate import  evaluate_model
from src.rf_predict import rf_predict_hate

st.title(" HATE-SPEECH DETECTION MODEL ")
#model=st.selectbox(" CHOOSE MODEL ",["LOGISTIC REGRESSION","SVM","RANDOM FOREST"])
user_input = st.text_area("Enter Text")
#st.write("CHOOSE MODEL ")
selected_model=st.radio("SELECTED MODEL",("LOGISTIC REGRESSION","SVM","RANDOM FOREST"))



 
if st.button("Predict") :

    #response=requests.get("http://127.0.0.1:8000/predict",params={"text":user_input,"model_type":selected_model})
    response=requests.get("https://hate-speech-detection-using-nlp-1.onrender.com/predict",params={"text":user_input,"model_type":selected_model})
    

    print(response.status_code)
    print(response.text)


    result=response.json()
    #result=rf_predict_hate(user_input)
    if selected_model == "LOGISTIC REGRESSION":
        with open("models/accuracy.txt","r") as f:
            accuracy=float(f.read())
        
        
    elif selected_model=="RANDOM FOREST":
        with open("models/rf_accuracy.txt","r") as f:
            accuracy=float(f.read())

    elif selected_model == "SVM":
        with open("models/svm_accuracy.txt","r") as f:
            accuracy = float(f.read())


    st.success(f"Prediction: {result['prediction']}\n\nThe Accuracy is {accuracy*100:.2f}%\n\nThe Confidence level is {result['confidence']}%"
)
    


