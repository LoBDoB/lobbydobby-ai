import joblib


class Classification():
    def predict(self, x):
        loaded_model = joblib.load('../rf_model.pkl')
        pred = loaded_model.predict(x)

        return pred