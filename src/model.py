from sklearn.tree import DecisionTreeClassifier
import joblib


class TitanicModel:
    def __init__(self):
        self.model = DecisionTreeClassifier(max_depth=3, random_state=42)

    def train(self, x_train, y_train):
        """Train the model and evaluate performance"""
        self.model.fit(x_train, y_train)

    def predict(self, user_input):
        """Make prediction for user input"""
        prediction = self.model.predict(user_input)
        probability = self.model.predict_proba(user_input)[0][1]
        return prediction[0], probability

    def save_model(self, path):
        """Save the trained model"""
        joblib.dump(self.model, path)

    def load_model(self, path):
        """Load a trained model"""
        self.model = joblib.load(path)
