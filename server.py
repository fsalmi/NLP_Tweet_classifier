# Importing the libraries
from flask import render_template,url_for,request, redirect
import numpy as np
from flask import Flask, request, jsonify
import pickle,json


app = Flask(__name__)
model = pickle.load(open('model.pkl','rb'))

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict(): 
    if request.method == 'POST':
        message = request.form['message']
        message=[message]
        print(message)
        #data = [stemmer.lemmatize(word) for word in message]
        my_prediction=model.predict(message)	
        #my_prediction=int(str(my_prediction))
        print(my_prediction)
        return render_template('result.html',prediction = my_prediction[0])
        return redirect('/')
    



if __name__ == '__main__':
	app.run(debug=True, host='0.0.0.0', port=12345)

