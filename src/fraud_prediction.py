import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import logging

logger = logging.getLogger(__name__)


class FraudPredictionModel:
    def __init__(self):
        self.model = None

    def train(self, X, y):
        """
        Train the fraud prediction model.

        :param X: Feature matrix
        :param y: Target vector
        """
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X_train, y_train)

        # Evaluate the model
        y_pred = self.model.predict(X_test)
        logger.info("Model Performance:\n" + classification_report(y_test, y_pred))

    def predict(self, incident):
        """
        Predict the probability of fraud for a given incident.

        :param incident: Dictionary containing incident details
        :return: Dictionary with fraud prediction results
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        # Extract features from the incident
        features = self.extract_features(incident)

        # Make prediction
        fraud_probability = self.model.predict_proba([features])[0][1]

        return {
            'fraud_probability': float(fraud_probability),
            'is_likely_fraud': fraud_probability > 0.5
        }

    def extract_features(self, incident):
        """
        Extract relevant features from the incident for prediction.
        This is a placeholder and should be implemented based on your specific requirements.

        :param incident: Dictionary containing incident details
        :return: List of extracted features
        """
        # Placeholder implementation - replace with actual feature extraction logic
        return [
            len(incident.get('description', '')),
            incident.get('severity', 0),
            # Add more relevant features here
        ]

    def save_model(self, filename):
        """Save the trained model to a file."""
        if self.model is None:
            raise ValueError("No trained model to save.")
        joblib.dump(self.model, filename)
        logger.info(f"Model saved to {filename}")

    def load_model(self, filename):
        """Load a trained model from a file."""
        self.model = joblib.load(filename)
        logger.info(f"Model loaded from {filename}")


# Example usage
if __name__ == "__main__":
    # This is a dummy dataset for demonstration. Replace with your actual dataset.
    X = np.random.rand(1000, 10)  # 1000 samples, 10 features
    y = np.random.randint(2, size=1000)  # Binary classification

    model = FraudPredictionModel()
    model.train(X, y)

    # Example prediction
    incident = {
        'description': 'Unusual login activity detected',
        'severity': 8
    }
    prediction = model.predict(incident)
    print(f"Fraud Prediction: {prediction}")

    # Save the model
    model.save_model('fraud_model.joblib')
