import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download and load the trained model
model_path = hf_hub_download(repo_id="ggoel1991/tourism_project", filename="best_tourism_model_v1.joblib")
model = joblib.load(model_path)

# Streamlit UI
st.title("Wellness Tourisim Prediction")
st.write("""
This predicts whether a customer will purchase the newly introduced Wellness Tourism Package before contacting them based on the
parameters like Age, Duration of Pitch, Monthly Income,etc.
Please enter the app details below to get a revenue prediction.
""")

# User input
age_category = st.selectbox("Age Category", ["Teen", "Young Adult", "Adult", "Middle Age", "Senior"])
duration_category= st.selectbox("Duration of Pitch", ["Very Short", "Short","Medium","Long","Very Long"])
income_category= st.selectbox("Income Category", ["Low Less than 10K", "Lower-Middle Greator than 10K  and Less than 20K", "Middle Greator than 20K  and Less than 40K", "Upper-Middle Greator than 40K  and Less than 70K", "High Greator than 70K"])
typeof_contact = st.selectbox("Type of Contact", ["Company Invited", "Self Contacted"])
city_tier = st.selectbox("City Tier", [1, 2, 3])
occupation = st.selectbox("Occupation", ["Free Lancer", "Small Business", "Large Business", "Salaried"])
gender = st.selectbox("Gender", ["Male", "Female"])
number_of_person_visiting = st.number_input("Number of Persons Visiting", min_value=1, max_value=10, value=2)
number_of_followups = st.number_input("Number of Followups", min_value=0, max_value=10, value=2)
product_pitched = st.selectbox("Product Pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"])
preferred_property_star = st.selectbox("Preferred Property Star", [3, 4, 5])
marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
number_of_trips = st.number_input("Number of Trips", min_value=0, max_value=20, value=1)
passport = st.selectbox("Passport", ["0", "1"])
pitch_satisfaction_score = st.slider("Pitch Satisfaction Score", min_value=1, max_value=5, value=3)
own_car = st.selectbox("Own Car", ["1", "0"])
number_of_person_visiting = st.number_input("Number of Persons Visiting", min_value=0, max_value=10, value=2)
designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])




# Assemble input into DataFrame
input_data = pd.DataFrame([{
    'Age_cat': age_category,
    'Duration_cat': duration_category,
    'Income_cat': income_category,
    'TypeofContact': typeof_contact,
    'CityTier': city_tier,
    'Occupation': occupation,
    'Gender': gender,
    'ProductPitched': product_pitched,
    'MaritalStatus': marital_status,
    'Designation': designation,
    'PreferredPropertyStar': preferred_property_star,
    'OwnCar': own_car,
    'Passport': passport,
    'PitchSatisfactionScore': pitch_satisfaction_score,
    'NumberOfPersonVisiting': number_of_person_visiting,
    'NumberOfFollowups': number_of_followups,
    'NumberOfTrips': number_of_trips
}])

if st.button("Predict"):
    prediction = model.predict(input_data)[0]

    st.subheader("Prediction Result:")

    if prediction == 0:
        st.error("Prediction says: **No Travel Package Taken**")
    else:
        st.success("Prediction says: **Travel Package Taken**")
