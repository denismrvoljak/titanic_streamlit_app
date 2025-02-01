import streamlit as st
import time
import os
import pandas as pd
from src.data_preprocessing import TitanicPreprocessor, InputPreprocessor
from src.model import TitanicModel
from src.config import TICKET_OPTIONS


def display_ticket(
    name, gender, age, embarkation, family_members, ticket_class, ticket_type, price
):
    st.markdown(
        f"""
        ---
        ## 🎟️ Titanic Boarding Pass  
        **Passenger:** {name}  
        **Gender:** {gender}  
        **Age:** {age}  
        **Embarkation Port:** {embarkation}  
        **Family Members Traveling:** {family_members}  
        **Ticket Class:** {ticket_class}  
        **Ticket Type:** {ticket_type}  
        **Price:** ${price}  
        ---
        """
    )


def main():
    st.title("🚢 Titanic Ticket Booking & Survival Prediction")

    st.markdown(
        "### Book your ticket for the Titanic! Will you survive? Let's find out... 🎟️"
    )

    # Initialize preprocessor and model
    preprocessor = TitanicPreprocessor()
    model = TitanicModel()

    @st.cache_data
    def load_data(file_path):
        root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        path = os.path.join(root_dir, file_path)
        data = pd.read_csv(path)
        return data

    data = load_data("titanic_streamlit_app/data/raw/train.csv")
    processed_data = preprocessor.transform(data)

    x_train = processed_data.drop(columns=["survived"])
    y_train = processed_data["survived"]

    model.train(x_train, y_train)

    # Initialize session state for purchased ticket
    if "ticket_purchased" not in st.session_state:
        st.session_state.ticket_purchased = False

    # Only show the form if no ticket has been purchased
    if not st.session_state.ticket_purchased:
        # Ticket class selection outside the form to allow immediate updates
        ticket_class = st.selectbox(
            "🎟️ Select Ticket Class", [1, 2, 3], key="ticket_class_select"
        )

        # Get available tickets for the selected class
        available_tickets = TICKET_OPTIONS[ticket_class]
        ticket_labels = [
            f"**{option[0]}** - 💰 ${option[1]}" for option in available_tickets
        ]

        # Ticket selection outside the form
        selected_ticket_index = st.radio(
            "🎫 Choose Your Ticket",
            ticket_labels,
            index=0,
            key=f"ticket_radio_{ticket_class}",
        )

        # Extract ticket type and price
        ticket_type, price = available_tickets[
            ticket_labels.index(selected_ticket_index)
        ]

        st.info(f"✅ You selected: **{ticket_type}** Ticket for **${price}**")

        with st.form("passenger_info"):
            name = st.text_input("📝 Your Name", placeholder="John Doe")

            col1, col2 = st.columns(2)
            with col1:
                gender = st.radio("🚻 Gender", ["Male", "Female"])
                age = st.number_input("🎂 Age", min_value=0, max_value=100, value=30)

            with col2:
                embarkation = st.selectbox(
                    "⛵ Port of Embarkation", ["Cherbourg", "Queenstown", "Southampton"]
                )
                family_members = st.number_input(
                    "👨‍👩‍👧‍👦 Family Members Traveling With You",
                    min_value=0,
                    max_value=10,
                    value=0,
                )

            # Ticket Purchase Button
            buy_ticket = st.form_submit_button("🎫 Buy Ticket")

        if buy_ticket:
            # Save all information in session state
            st.session_state.update(
                {
                    "name": name,
                    "gender": gender,
                    "age": age,
                    "embarkation": embarkation,
                    "family_members": family_members,
                    "ticket_class": ticket_class,
                    "ticket_type": ticket_type,
                    "ticket_price": price,
                    "ticket_purchased": True,
                }
            )

            # Simulating ticket generation
            with st.spinner("Generating your Titanic Ticket... 🎟️"):
                time.sleep(2)
            st.success("✅ Ticket Successfully Purchased!")
            st.rerun()

    # If ticket is purchased, display the ticket and prediction section
    if st.session_state.ticket_purchased:
        display_ticket(
            st.session_state.name,
            st.session_state.gender,
            st.session_state.age,
            st.session_state.embarkation,
            st.session_state.family_members,
            st.session_state.ticket_class,
            st.session_state.ticket_type,
            st.session_state.ticket_price,
        )

        if st.button("🚢 Board Titanic & Predict My Fate"):
            with st.spinner(
                "The Titanic is sailing... 🌊 Checking your survival fate..."
            ):
                time.sleep(4)  # Suspense effect

            # Prepare input & predict
            user_input = InputPreprocessor.prepare_user_input(
                st.session_state.name,
                st.session_state.gender,
                st.session_state.embarkation,
                st.session_state.family_members,
                st.session_state.age,
                st.session_state.ticket_class,
                st.session_state.ticket_type,
            )
            prediction, probability = model.predict(user_input)

            # **Reveal survival fate**
            if prediction == 1:
                st.balloons()
                st.success(
                    f"🎉 **Congratulations {st.session_state.name}! You survived the Titanic!** 🚢 "
                    f"(_Survival Probability: {probability:.2%}_)"
                )
            else:
                st.error(
                    f"💀 **Unfortunately, {st.session_state.name}, you did not survive...** "
                    f"(_Survival Probability: {probability:.2%}_)"
                )


if __name__ == "__main__":
    main()
