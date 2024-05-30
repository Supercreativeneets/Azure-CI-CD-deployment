import os
import sys

from flask import Flask, render_template, request

import pandas as pd
from src.pipeline.predict_pipeline import PredictPipeline

application = Flask(__name__)

app = application

def get_dates():
    test_path = os.path.join('artifacts', 'test.csv')
    test_df = pd.read_csv(test_path, index_col=None, header=0)
    values = test_df.loc[24:, 'date']
    dates = values.tolist()
    return dates

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'GET':
        dates = get_dates()
        return render_template('index.html', dates=dates)
    elif request.method == 'POST':
        selected_date = request.form['date']
        predict_pipeline = PredictPipeline()
        prediction = predict_pipeline.predict(selected_date)
        return render_template('result.html', result=prediction)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)