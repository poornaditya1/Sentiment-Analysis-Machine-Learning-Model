import streamlit as st
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix,classification_report


df = pd.read_csv('amazon review.csv')
df1 = df.dropna(subset=['reviews.text'])
df1 = df1.dropna(subset=['reviews.rating'])
X = df1.iloc[:,16].values
y = df1.iloc[:,14].values
import numpy as np
Y=[0] * (len(y))
for i in range(0,(len(y))):
  if y[i]==1 or y[i]==2 or y[i]==3:
    Y[i] = 0              # Negative sentiments
  else:
    Y[i] = 1              # Positive sentiments
text_model = Pipeline([('tfidf',TfidfVectorizer(binary = True,max_df=0.611111111111111,norm = 'l2')),('model',MultinomialNB(alpha = 0.058,fit_prior=True))])

text_model.fit(X,Y)

def u_in():
  text = st.text_input("Enter your review: ")
  a = []
  a.append(text)
  return a

st.title("ML Project")
df = u_in()
pred = text_model.predict(df)
if pred == 0:
  st.write("Negative Sentiment")
else:
  st.write("Positive Sentiment")
st.footer("Made by Poornaditya Mishra")
