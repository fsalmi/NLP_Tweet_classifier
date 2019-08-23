# Importing the libraries
import numpy as np
import re
import nltk
import pandas as pd
from sklearn.datasets import load_files
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from flask import render_template,url_for,request




import pickle
import requests
import json


#Importing the dataset 
tweetdata=pd.read_csv('tweet_data.csv')

#Clean the text
def  clean_text(tweets):
    tweets["tweet"] = tweets["tweet"].str.lower()
    tweets["tweet"] = tweets["tweet"].apply(lambda elem: re.sub(r"(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?|rt|\d+|amp", "", elem))  
    tweets["tweet"] = tweets["tweet"].apply(lambda elem: re.sub(r'\s+|\s+[a-zA-Z]\s+',' ', elem))  

    return tweets
clean_text(tweetdata)
X=tweetdata.tweet
y=tweetdata[['class']]
    

#Lemmatize 
stemmer = WordNetLemmatizer()
X = [stemmer.lemmatize(word) for word in X]

#Creating the pipeline that will regroup the different steps 
pipeline_sgd = Pipeline([
    ('vect', CountVectorizer(max_features=1500, stop_words=stopwords.words('english'))),
    ('tfidf',  TfidfTransformer()),
    ('classifier', RandomForestClassifier(n_estimators=50, random_state=0)),
])

#Split the data into train and test     
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

#Use the model to predict the label of test tweets 
model = pipeline_sgd.fit(X_train, y_train)
y_pred = model.predict(X_test)

#Print the metrics results 
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
print(accuracy_score(y_train, model.predict(X_train)))
print(accuracy_score(y_test, y_pred))

# Saving model to disk
pickle.dump(model, open('model.pkl','wb'))
# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))

#print(model.predict([[1.8]]))