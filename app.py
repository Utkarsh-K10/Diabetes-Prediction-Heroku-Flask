import numpy as np
import pickle
from flask import Flask, request, render_template, jsonify

app = Flask(__name__)

model = pickle.load(open('diabetes-preediction-model.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    glucose = float(request.form['glucose'])
    bp = float(request.form['bloodpressure'])
    sknthic = float(request.form['skinthickness'])
    insulin = float(request.form['insulin'])
    bmi = float(request.form['bmi'])
    dpf = float(request.form['dpf'])
    age = int(request.form['age'])
    pregn = int(request.form['pregnancies'])

    params = [glucose, bp, sknthic, insulin, bmi, dpf, age, pregn]
    user_input = [np.array(params)]
    prediction = model.predict(user_input)
    return render_template('index.html', final_output=prediction)


if __name__ == "__main__":
    app.run(debug=True)

#features
#Glucose    BP  SkinThickness   Insulin BMI DPF Age Pregnancies 