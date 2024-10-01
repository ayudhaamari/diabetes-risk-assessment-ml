# Import necessary libraries
import streamlit as st  # For creating the web application interface
import pandas as pd  # For data manipulation and analysis
import pickle  # For loading the trained machine learning model

# Load the trained model from the pickle file
# This model has been previously trained and saved
with open('tuned_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Function to convert age to categorical value
# This is necessary because the model expects age as a categorical input
def age_to_category(age):
    if 18 <= age <= 24:
        return 1
    elif 25 <= age <= 29:
        return 2
    elif 30 <= age <= 34:
        return 3
    elif 35 <= age <= 39:
        return 4
    elif 40 <= age <= 44:
        return 5
    elif 45 <= age <= 49:
        return 6
    elif 50 <= age <= 54:
        return 7
    elif 55 <= age <= 59:
        return 8
    elif 60 <= age <= 64:
        return 9
    elif 65 <= age <= 69:
        return 10
    elif 70 <= age <= 74:
        return 11
    elif 75 <= age <= 79:
        return 12
    else:
        return 13

# Function to convert education level to categorical value
# This mapping is based on the original dataset's encoding
def education_to_category(education):
    education_map = {
        "Never / kindergarten": 1,
        "Elementary school": 2,
        "Junior school": 3,
        "High school Graduate": 4,
        "College": 5,
        "College graduate": 6
    }
    return education_map.get(education, 1)  # Default to 1 if education level is not found

# Function to convert income level to categorical value
# This mapping is based on the original dataset's encoding
def income_to_category(income):
    income_map = {
        "Less than $10,000": 1,
        "$10,000 to less than $15,000": 2,
        "$15,000 to less than $20,000": 3,
        "$20,000 to less than $25,000": 4,
        "$25,000 to less than $35,000": 5,
        "$35,000 to less than $50,000": 6,
        "$50,000 to less than $75,000": 7,
        "$75,000 or more": 8
    }
    return income_map.get(income, 1)  # Default to 1 if income level is not found

# Main function to run the Streamlit app
def run():
    # Set the title of the web application
    st.title('🩺 Diabetes Risk Assessment')

    # Sidebar for additional information and links
    st.sidebar.markdown("---")
    st.sidebar.subheader("ℹ️ Input Information")
    st.sidebar.info(
        "Adjust the sliders and select options to input your health information. "
        "The model will use this data to assess your diabetes risk."
    )
    st.sidebar.markdown("---")
    st.sidebar.subheader("🔗 Useful Links")
    st.sidebar.markdown("[CDC Diabetes Info](https://www.cdc.gov/diabetes/index.html)")
    st.sidebar.markdown("[WHO Diabetes Overview](https://www.who.int/health-topics/diabetes)")

    # Display an image in the center of the page
    col1, col2, col3 = st.columns([1,3,1])
    with col2:
        st.image("images/diabetes-ml.jpg", 
                 caption="Source: MeriSKILL",
                 use_column_width=True)

    st.write("Please answer the following questions about your health and lifestyle:")

    st.markdown("---")
    st.subheader("📋 Health and Lifestyle Questionnaire")
    
    # Collect user input for various health indicators
    # Each input is accompanied by an emoji for better visual representation
    HighBP = 1 if st.radio("🩸 Do you have high blood pressure?", ["No", "Yes"]) == "Yes" else 0
    st.write("")

    HighChol = 1 if st.radio("🍔 Do you have high cholesterol?", ["No", "Yes"]) == "Yes" else 0
    st.write("")

    CholCheck = 1 if st.radio("🩺 Have you had a cholesterol check in the past 5 years?", ["No", "Yes"]) == "Yes" else 0
    st.write("")

    BMI = st.slider('⚖️ What is your BMI?', min_value=10.0, max_value=60.0, value=25.0, step=0.1)
    st.write("")

    Smoker = 1 if st.radio("🚬 Are you a smoker?", ["No", "Yes"]) == "Yes" else 0
    st.write("")

    Stroke = 1 if st.radio("🧠 Have you ever had a stroke?", ["No", "Yes"]) == "Yes" else 0
    st.write("")

    HeartDiseaseorAttack = 1 if st.radio("❤️ Do you have heart disease or have you had a heart attack?", ["No", "Yes"]) == "Yes" else 0
    st.write("")

    PhysActivity = 1 if st.radio("🏃‍♂️ Have you engaged in physical activity in the past 30 days?", ["No", "Yes"]) == "Yes" else 0
    st.write("")

    Fruits = 1 if st.radio("🍎 Do you consume fruit 1 or more times per day?", ["No", "Yes"]) == "Yes" else 0
    st.write("")

    Veggies = 1 if st.radio("🥕 Do you consume vegetables 1 or more times per day?", ["No", "Yes"]) == "Yes" else 0
    st.write("")

    HvyAlcoholConsump = 1 if st.radio("🍺 Do you consume heavy amounts of alcohol?", ["No", "Yes"]) == "Yes" else 0
    st.write("")

    AnyHealthcare = 1 if st.radio("🏥 Do you have any form of healthcare coverage?", ["No", "Yes"]) == "Yes" else 0
    st.write("")

    NoDocbcCost = 1 if st.radio("💰 In the past 12 months, was there a time when you needed to see a doctor but could not because of cost?", ["No", "Yes"]) == "Yes" else 0
    st.write("")

    GenHlth = st.selectbox('🌟 How would you rate your general health?', [1, 2, 3, 4, 5], format_func=lambda x: ["Excellent", "Very Good", "Good", "Fair", "Poor"][x-1])
    st.write("")

    MentHlth = st.slider('🧘‍♀️ In the past 30 days, for how many days was your mental health not good?', min_value=0, max_value=30, value=0, step=1)
    st.write("")

    PhysHlth = st.slider('🤒 In the past 30 days, for how many days was your physical health not good?', min_value=0, max_value=30, value=0, step=1)
    st.write("")

    DiffWalk = 1 if st.radio("🚶‍♂️ Do you have serious difficulty walking or climbing stairs?", ["No", "Yes"]) == "Yes" else 0
    st.write("")

    Sex = 1 if st.radio("👥 What is your gender?", ["Female", "Male"]) == "Male" else 0
    st.write("")

    Age = st.number_input('🎂 What is your age?', min_value=1, max_value=150, value=30, step=1)
    Age_category = age_to_category(Age)  # Convert age to category
    st.write("")

    Education_options = [
        "Never / kindergarten",
        "Elementary school",
        "Junior school",
        "High school Graduate",
        "College",
        "College graduate"
    ]
    Education = st.selectbox('🎓 What is your highest level of education?', Education_options)
    Education_category = education_to_category(Education)  # Convert education to category
    st.write("")

    Income_options = [
        "Less than $10,000",
        "$10,000 to less than $15,000",
        "$15,000 to less than $20,000",
        "$20,000 to less than $25,000",
        "$25,000 to less than $35,000",
        "$35,000 to less than $50,000",
        "$50,000 to less than $75,000",
        "$75,000 or more"
    ]
    Income = st.selectbox('💵 What is your income level?', Income_options)
    Income_category = income_to_category(Income)  # Convert income to category
    st.markdown("---")

    # Button to trigger the risk assessment
    if st.button('Assess Diabetes Risk'):
        # Create a dataframe with the user input
        # This dataframe will be used as input for the prediction model
        input_data = pd.DataFrame({
            'HighBP': [HighBP], 'HighChol': [HighChol], 'CholCheck': [CholCheck],
            'BMI': [BMI], 'Smoker': [Smoker], 'Stroke': [Stroke],
            'HeartDiseaseorAttack': [HeartDiseaseorAttack], 'PhysActivity': [PhysActivity],
            'Fruits': [Fruits], 'Veggies': [Veggies], 'HvyAlcoholConsump': [HvyAlcoholConsump],
            'AnyHealthcare': [AnyHealthcare], 'NoDocbcCost': [NoDocbcCost],
            'GenHlth': [GenHlth], 'MentHlth': [MentHlth], 'PhysHlth': [PhysHlth],
            'DiffWalk': [DiffWalk], 'Sex': [Sex], 'Age': [Age_category],
            'Education': [Education_category], 'Income': [Income_category]
        })

        # Make prediction using the loaded model
        prediction = model.predict(input_data)

        # Display results based on the prediction
        st.subheader('Assessment Result:')
        if prediction[0] == 0:
            # If the prediction is 0, the risk is lower
            st.success('Based on the provided information, you are at a **lower risk** of diabetes.')
            st.image("https://static.vecteezy.com/system/resources/previews/015/887/580/original/healthy-mental-man-icon-outline-style-vector.jpg", width=300)
            st.write("While your risk is lower, it's always good to maintain a healthy lifestyle.")
            st.write("Here are some tips to stay healthy:")
            tips = [
                "Maintain a balanced diet",
                "Exercise regularly",
                "Get regular check-ups"
            ]
            for tip in tips:
                st.markdown(f"- {tip}")
        else:
            # If the prediction is 1, the risk is higher
            st.warning('Based on the provided information, you may be at a **higher risk** of diabetes.')
            st.image("https://cdn-icons-png.flaticon.com/512/6762/6762066.png", width=300)
            st.write("Don't worry! Here are some steps you can take:")
            steps = [
                "Consult with a healthcare professional for a thorough evaluation",
                "Monitor your blood sugar levels regularly",
                "Adopt a healthy, balanced diet",
                "Maintain a healthy weight",
            ]
            for step in steps:
                st.markdown(f"- {step}")
        
        # Display a disclaimer about the assessment
        st.info('Please note that this is a preliminary assessment based on general factors. For a definitive diagnosis, please consult with a healthcare professional.')
        
        # Display additional resources
        st.markdown("---")
        st.subheader("Additional Resources")
        st.write("Here are some helpful resources for managing diabetes risk:")
        resources = {
            "American Diabetes Association": "https://www.diabetes.org/",
            "CDC Diabetes Resources": "https://www.cdc.gov/diabetes/index.html",
            "National Institute of Diabetes and Digestive and Kidney Diseases": "https://www.niddk.nih.gov/health-information/diabetes",
            "World Health Organization - Diabetes": "https://www.who.int/health-topics/diabetes"
        }
        for name, link in resources.items():
            st.markdown(f"- [{name}]({link})")

# Run the Streamlit app if this script is executed directly
if __name__ == '__main__':
    run()