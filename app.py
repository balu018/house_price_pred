from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    val1 = request.form['lot_area']
   
    val3 = request.form['overall_quality']
    val4 = request.form['yr_built']
    

    val7 = request.form['living_area']
    val8 = request.form['bed_rooms']
    

    try:
        
        
        
        arr = np.array([val1, val3, val4, val7, val8])
        arr = arr.astype(np.float64)
        arr = arr.reshape(1, -1)  
        pred = model.predict(arr)

        return render_template('index.html', data=int(pred))
    except ValueError as e:
        # Handle unknown category error
        error_message = f"Error: {str(e)}"
        return render_template('index.html', data=error_message)


if __name__ == '__main__':
    app.run(debug=True)
