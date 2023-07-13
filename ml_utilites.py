import joblib
import numpy as np
import pandas as pd

MODEL_DIR = "ml_pipeline/Weights/"


class Predictor:
    def __init__(self):
        self.model = joblib.load(f'{MODEL_DIR}model.pkl')
        self.preprocessor = joblib.load(f'{MODEL_DIR}preprocessor.pkl')
        self.features = None

    def process_features(self, year, make, model):
        input_features = pd.DataFrame({
            'make': [make],
            'model': [model],
            'age': [2023-int(year)]
        })
        self.features = self.preprocessor.transform(input_features)

    def get_prediction(self):
        prediction = self.model.predict(self.features)
        prediction = np.floor(prediction[-1]).astype(int)
        return abs(prediction)
