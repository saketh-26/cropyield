from Crop_Yield_prediction_Model import predict_yield as py, final_model
import numpy as np
from flask import Flask, request, render_template
app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':

        form = request.form
        area = float(form['area'])
        item = int(form['item'])
        year = int(form['year'])
        rainfall = float(form['rainfall'])
        #  gender = form['gender']
        pesticides = float(form['pesticides'])
        temp = float(form['temp'])
        p = []
        p += [area, item, year, rainfall, pesticides, temp]
        result = py([p], final_model)
        return render_template('index.html', res=str(result))
    return render_template('index.html', res="Fill the details and Click Submit")


app.run()
