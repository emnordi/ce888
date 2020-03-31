from flask import Flask, request, render_template
import os
import pickle

print(os.getcwd())
path = os.getcwd()

with open('Models/logistic_model.pkl', 'rb') as f:
    logistic = pickle.load(f)

with open('Models/RF_model.pkl', 'rb') as f:
    randomforest = pickle.load(f)

with open('Models/svm_clf_model.pkl', 'rb') as f:
    svm_model = pickle.load(f)


def get_predictions(age, sex, cp, testbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal):
    req_model = 'RandomForest'
    mylist = [age, sex, cp, testbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]
    mylist = [float(i) for i in mylist]
    vals = [mylist]

    if req_model == 'Logistic':
        #print(req_model)
        return logistic.predict(vals)[0]

    elif req_model == 'RandomForest':
        #print(req_model)
        return randomforest.predict(vals)[0]

    elif req_model == 'SVM':
        #print(req_model)
        return svm_model.predict(vals)[0]
    else:
        return "Cannot Predict"


app = Flask(__name__)

@app.route('/')
def my_form():
    return render_template('home.html')

@app.route('/', methods=['POST', 'GET'])
def my_form_post():
    age = request.form['age']
    sex = request.form['sex']
    cp = request.form['cp']
    testbps = request.form['testbps']
    chol = request.form['chol']
    fbs = request.form['fbs']
    restecg = request.form['restecg']
    thalach = request.form['thalach']
    exang = request.form['exang']
    oldpeak = request.form['oldpeak']
    slope = request.form['slope']
    ca = request.form['ca']
    thal = request.form['thal']

    target = get_predictions(age, sex, cp, testbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal)

    if target==1:
        sale_making = 'Person is likely to have heart disease'
    else:
        sale_making = 'Person is unlikely to have heart disease'

    return render_template('home.html', target = target, sale_making = sale_making)


if __name__ == "__main__":
    app.run(debug=True)