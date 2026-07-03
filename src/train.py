import pickle
import pandas as pd
from preprocess import clean_text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from evaluate import evaluate_model
from sklearn.metrics import confusion_matrix



df=pd.read_csv("data/kaggle_hate_speech_emoji.csv")
df["clean_tweet"]=df["tweet"].apply(clean_text)

X=df["clean_tweet"]
y=df["class"]
vectorizer= TfidfVectorizer(max_features=5000)
X=vectorizer.fit_transform(X)

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)

model=LogisticRegression(max_iter=1000,class_weight="balanced")

model.fit(X_train,y_train)
acc=evaluate_model(model,X_test,y_test)
with open("models/accuracy.txt","w") as f:
    f.write(str(acc))


y_pred = model.predict(X_test)

print(confusion_matrix(y_test, y_pred))
print(df["class"].value_counts())
print(df[["class","tweet"]].head(20))


with open("models/model.pkl","wb") as f:
    pickle.dump(model,f)

with open("models/vectorizer.pkl","wb") as f:
    pickle.dump(vectorizer,f)

print("\nModel saved Successfully.")
