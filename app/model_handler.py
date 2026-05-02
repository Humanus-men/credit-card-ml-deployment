import pickle
import numpy as np
import pandas as pd

class ModelHandler:
    def __init__(self, model_path, preprocessor_path="models/preprocessor.pkl"):
        with open(model_path, "rb") as f:
            self.model = pickle.load(f)
        with open(preprocessor_path, "rb") as f:
            self.preprocessor = pickle.load(f)

    def preprocess(self, data: dict) -> np.ndarray:
        # Приводим словарь к DataFrame, затем применяем сохранённый ColumnTransformer
        df = pd.DataFrame([data])
        return self.preprocessor.transform(df)

    def predict(self, data: dict) -> dict:
        features = self.preprocess(data)
        pred = int(self.model.predict(features)[0])
        proba = float(self.model.predict_proba(features)[0][1])
        return {"prediction": pred, "probability": proba}