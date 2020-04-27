import numpy as np


# Decision threshold for the regression output to convert to a binary classification
LOGISTIC_THRESHOLD = 0.5


class PretrainedRegression:
    """
    Pretrained logistic regression predictor

    The weights used to predict the output values are from the article titled
    Detecting Cyberbullying “Hotspots” on Twitter: A Predictive Analytics Approach from Ho et al. (2020).
    """

    def __init__(self):
        self.predicted_values = None

    @staticmethod
    def predict_row(x):
        return -4.270 + 0.007 * x[0] + 0.127 * x[1] + 0.006 * x[2] - 0.235 * x[3] + 0.281 * x[4] + 0.296 * x[5] + 0.428 * x[6] + 0.248 * x[7] + 0.496 * x[8] + 0.196 * x[9]

    def predict(self, X):
        predictions = []
        for row in X:
            prediction = self.predict_row(row)
            if prediction > LOGISTIC_THRESHOLD:
                predictions.append(1)
            else:
                predictions.append(0)
        self.predicted_values = np.array(predictions)
        return self.predicted_values

    def score(self, X=None, actual_values=None):
        if self.predicted_values is None:
            self.predict(X)

        correct_guesses = 0
        
        for (prediction, real_value) in zip(self.predicted_values, actual_values):
           if prediction == real_value:
               correct_guesses += 1 
        return correct_guesses / len(actual_values)
