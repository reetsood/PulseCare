import pandas as pd
import numpy as np
import pickle
from flask import Flask, request, render_template

app= Flask(__name__, static_url_path='/Flask/static')
model=pickle.load(open('predictive_pulse_model.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/check')
def check():
    return render_template('check.html')

@app.route('/details')
def details():
    show = request.args.get('show', 'form')  # default is 'form'
    show_info = True if show == 'info' else False
    return render_template('details.html', show_info=show_info)



@app.route('/predict', methods=["POST"])
def predict():

    gender_map = {'Male': 0, 'Female': 1}
    binary_map = {'No': 0, 'Yes': 1}
    severity_map = {'Mild': 0, 'Moderate': 1, 'Severe': 2}
    age_map = {'18-34':0,'35-50':1,'51-64':2,'65+':3}
    diastolic_map={'81-90':0,'91-100':1,'100+':2}
    systolic_map={'111-120':0,'121-130':1,'130+':2}
    whendiagoused_map={'<1year':0,'1-5years':1,'>5years':2}

    Gender = gender_map[request.form['Gender']]
    Age = age_map[request.form['Age']]
    History = binary_map[request.form['History']]
    Patient = binary_map[request.form['Patient']]
    TakeMedication = binary_map[request.form['TakeMedication']]
    Severity = severity_map[request.form['Severity']]
    BreathShortness = binary_map[request.form['BreathShortness']]
    VisualChange = binary_map[request.form['VisualChanges']]
    NoseBleeding = binary_map[request.form['NoseBleeding']]
    WhenDiagnoused = whendiagoused_map[request.form['Whendiagnoused']]
    Systolic = systolic_map[request.form['Systolic']]
    Diastolic = diastolic_map[request.form['Diastolic']]
    ControlledDiet = binary_map[request.form['ControlledDiet']]

    features_values = np.array([[Gender, Age,History, Patient,TakeMedication, Severity, BreathShortness, VisualChange, NoseBleeding,
                                 WhenDiagnoused, Systolic, Diastolic, ControlledDiet]])

    df = pd.DataFrame(features_values, columns=['Gender', 'Age','History', 'Patient','TakeMedication','Severity', 'BreathShortness',
                                                'VisualChanges', 'NoseBleeding', 'Whendiagnoused',
                                                'Systolic', 'Diastolic', 'ControlledDiet'])
    print(df)

    prediction = model.predict(df)
    print(prediction[0])
    if prediction[0] == 0:
        result = "NORMAL"
    elif prediction[0] == 1:
        result = "HYPERTENSION (Stage-1)"
    elif prediction[0] == 2:
        result = "HYPERTENSION (Stage-2)"
    else:
        result = "HYPERTENSIVE CRISIS"

    print(result)
    text = 'Your Blood Pressure stage is: '
    return render_template('prediction.html', prediction_text=text+result, category=result)


if __name__ == '__main__':
    app.run(debug=True)