from flask import Flask, request, render_template
import os
import sys
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)
app = application

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        # Create CustomData instance
        custom_data = CustomData(
            age=request.form.get('age'),
            sex=request.form.get('sex'),
            bmi=request.form.get('bmi'),
            children=request.form.get('children'),
            smoker=request.form.get('smoker'),
            region=request.form.get('region')
        )

        #Convert to dataframe
        input_data = custom_data.get_data_as_dataframe()

        #Predict
        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(input_data)

        return render_template('home.html', results=results[0])

if __name__=="__main__":
    app.run(host="0.0.0.0", debug=True)       