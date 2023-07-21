from ml_utilites import Predictor
from flask import Flask, render_template, request

app = Flask(__name__)
predictor = Predictor()


@app.route('/')
def search_page():
    return render_template('search.html')


@app.route('/results', methods=['POST'])
def results_page():
    year = request.form['year']
    make = request.form['make']
    model = request.form['model']

    predictor.process_features(year, make, model)
    prediction = predictor.get_prediction()

    return render_template('results.html', year=year, make=make, model=model, price=prediction)


if __name__ == '__main__':
    app.run(host='0.0.0.0')
