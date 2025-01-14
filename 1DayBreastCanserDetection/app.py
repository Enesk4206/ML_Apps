from flask import Flask, request, render_template
import pandas
import numpy as np 
import pickle

model = pickle.load(open("model.pkl", 'rb'))

app = Flask(__name__)

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    features = request.form['feature']
    features_lst = features.split(',')
    np_features = np.asarray(features_lst,dtype=float)
    pred = model.predict(np_features.reshape(1,-1))
    
    output = ["cancrous" if pred[0] == 1 else "not cancrous"]
    
    return render_template('index.html', message=output)


if __name__ == "__main__":
    app.run(debug=True)