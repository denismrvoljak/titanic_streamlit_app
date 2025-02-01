# app.py
import streamlit as st
from src.data_preprocessing import TitanicPreprocessor
from src.data_preprocessing import InputPreprocessor
from src.model import TitanicModel
from src.config import TICKET_PRICES
import os
import pandas as pd


def main():
    st.title("🚢 Titanic Survival Predictor")

    # Initialize preprocessor and model
    preprocessor = TitanicPreprocessor()
    model = TitanicModel()

    def load_data(file_path):
        root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        path = os.path.join(root_dir, file_path)
        data = pd.read_csv(path)
        return data

    data = load_data("titanic_streamlit_app/data/raw/train.csv")

    processed_data = preprocessor.transform(data)

    x_train = processed_data.drop(columns=["survived"])  # Features
    y_train = processed_data["survived"]  # Target variable

    model.train(x_train, y_train)

    # User inputs
    with st.form("passenger_info"):
        name = st.text_input("Your Name")

        col1, col2 = st.columns(2)
        with col1:
            gender = st.radio("Gender", ["Male", "Female"])
            age = st.number_input("Age", min_value=0, max_value=100, value=30)

        with col2:
            embarkation = st.selectbox(
                "Port of Embarkation", ["Cherbourg", "Queenstown", "Southampton"]
            )
            family_members = st.number_input(
                "Number of Family Members Traveling with You",
                min_value=0,
                max_value=10,
                value=0,
            )

        # Ticket selection
        ticket_class = st.selectbox("Ticket Class", [1, 2, 3])
        ticket_type = st.selectbox(
            "Ticket Type", ["Budget", "Standard", "Premium", "Luxury"]
        )

        # Show ticket price
        price = TICKET_PRICES.get((ticket_class, ticket_type), 0)
        st.write(f"Ticket Price: ${price}")

        submitted = st.form_submit_button("Predict My Fate")

    if submitted:
        # Instantiate InputPreprocessor
        user_input = InputPreprocessor.prepare_user_input(
            name, gender, embarkation, family_members, age, ticket_class, ticket_type
        )

        # Make prediction
        prediction, probability = model.predict(user_input)

        # Display result
        if prediction == 1:
            st.success(
                f"Congratulations {name}! You would likely survive! (Probability: {probability:.2%})"
            )
        else:
            st.error(
                f"Unfortunately {name}, you might not survive... (Probability: {probability:.2%})"
            )


if __name__ == "__main__":
    main()
