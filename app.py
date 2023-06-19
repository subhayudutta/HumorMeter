import json
import pickle
from flask import Flask,request,app,jsonify,url_for,render_template
import numpy as np
import pandas as pd
import nltk
import tensorflow as tf
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import re
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
import tensorflow

app=Flask(__name__)

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        sample=request.form['text']
        ps = PorterStemmer()
        A = []
        review = re.sub('[^a-zA-Z]', ' ', sample)
        review = review.lower()
        review = review.split()    
        #review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
        review = ' '.join(review)
        A.append(review)

        sample_repr=[one_hot(words,5000)for words in A] 
        sent_length=20
        sample_repr=pad_sequences(sample_repr,padding='post',maxlen=sent_length)
    
        pickled_model=load_model('humorPred.h5',compile=False)
        ans=pickled_model.predict(sample_repr)
        ans = True if ans > 0.5 else False
    return render_template("index.html",hum="Humor Status - ",prediction_text=ans,thought='Your thoughts - ',inp=sample)


if __name__=="__main__":
    app.run(debug=True)