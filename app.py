import numpy as np
import pickle
from flask import Flask,request,render_template

app = Flask(__name__)
model = pickle.load(open('model.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods = ['POST'])
def predict():
    features = [float(x) for x in request.form.values()]
    final_features = np.array(features)
    final_features=final_features.reshape(1,-1)
    prediction = model.predict(final_features)

    if(prediction[0]==1):
        return render_template('index.html',prediction_text='You have a heart disease, imediately consult a doctor.')
    else:
        return render_template('index.html',prediction_text='You are fine.')

if __name__=="__main__":
    app.run(debug=True)
    