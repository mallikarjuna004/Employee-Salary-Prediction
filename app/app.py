"""
=========================================================
Employee Salary Prediction System
=========================================================
"""

import json
import os

import joblib
import pandas as pd
import streamlit as st

# ==========================================================
# Paths
# ==========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "model",
    "salary_pipeline.pkl"
)

METRICS_PATH = os.path.join(
    PROJECT_ROOT,
    "model",
    "metrics.json"
)

# ==========================================================
# Page Configuration
# ==========================================================

st.set_page_config(
    page_title="Employee Salary Prediction",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================================
# Custom CSS
# ==========================================================

st.markdown("""

<style>

.main{

    padding-top:20px;

}

.hero{

    background:linear-gradient(135deg,#0f172a,#1e3a8a);

    padding:25px;

    border-radius:15px;

    color:white;

    margin-bottom:25px;

}

.hero h1{

    margin:0;

}

.hero p{

    margin-top:10px;

    font-size:18px;

}

.metric-box{

    background:#f5f5f5;

    border-radius:12px;

    padding:12px;

    border:1px solid #e5e5e5;

}

footer{

    visibility:hidden;

}

#MainMenu{

    visibility:hidden;

}

</style>

""", unsafe_allow_html=True)

# ==========================================================
# Load Model
# ==========================================================

@st.cache_resource

def load_pipeline():

    return joblib.load(MODEL_PATH)


# ==========================================================
# Load Metrics
# ==========================================================

@st.cache_data

def load_metrics():

    with open(METRICS_PATH) as file:

        return json.load(file)


pipeline = load_pipeline()

metrics = load_metrics()

# ==========================================================
# Sidebar
# ==========================================================

with st.sidebar:

    st.title("📌 Project")

    st.divider()

    st.subheader("🤖 Model")

    st.success("Random Forest")

    st.subheader("📊 Accuracy")

    st.metric(

        "Accuracy",

        f"{metrics['accuracy']*100:.2f}%"

    )

    st.metric(

        "Precision",

        f"{metrics['precision']*100:.2f}%"

    )

    st.metric(

        "Recall",

        f"{metrics['recall']*100:.2f}%"

    )

    st.metric(

        "F1 Score",

        f"{metrics['f1_score']*100:.2f}%"

    )

    st.divider()

    st.subheader("📂 Dataset")

    st.info("Adult Income Dataset")

    st.subheader("🛠 Technology")

    st.write("✔ Python")

    st.write("✔ Pandas")

    st.write("✔ Scikit-Learn")

    st.write("✔ Streamlit")

    st.write("✔ Joblib")

    st.divider()

    st.caption("Developed by")

    st.caption("Vallamkonda mallikarjuna rao")

# ==========================================================
# Hero Section
# ==========================================================

st.markdown("""

<div class="hero">

<h1>💼 Employee Salary Prediction System</h1>

<p>
Predict whether an employee's annual income is <b>&gt; $50K</b> or <b>&le; $50K</b> using a Machine Learning model trained on the UCI Adult Income Dataset.
</p>

</div>

""", unsafe_allow_html=True)

# ==========================================================
# Top Metrics
# ==========================================================

col1,col2,col3=st.columns(3)

with col1:

    st.metric(

        "Train Samples",

        metrics["train_samples"]

    )

with col2:

    st.metric(

        "Test Samples",

        metrics["test_samples"]

    )

with col3:

    st.metric(

        "Accuracy",

        f"{metrics['accuracy']*100:.2f}%"

    )

st.divider()

# ==========================================================
# Employee Details
# ==========================================================

st.header("📝 Employee Details")

with st.form("prediction_form"):

    col1, col2 = st.columns(2)

    # -----------------------
    # Left Column
    # -----------------------

    with col1:

        age = st.slider(
            "Age",
            17,
            90,
            30
        )

        workclass = st.selectbox(
            "Workclass",
            [
                "Private",
                "Self-emp-not-inc",
                "Self-emp-inc",
                "Federal-gov",
                "Local-gov",
                "State-gov",
                "Without-pay",
                "Never-worked"
            ]
        )

        education = st.selectbox(
            "Education",
            [
                "Bachelors",
                "Masters",
                "Doctorate",
                "HS-grad",
                "Some-college",
                "Assoc-acdm",
                "Assoc-voc",
                "Prof-school",
                "11th",
                "10th",
                "9th",
                "7th-8th"
            ]
        )

        marital_status = st.selectbox(
            "Marital Status",
            [
                "Never-married",
                "Married-civ-spouse",
                "Divorced",
                "Separated",
                "Widowed"
            ]
        )

        occupation = st.selectbox(
            "Occupation",
            [
                "Tech-support",
                "Craft-repair",
                "Other-service",
                "Sales",
                "Exec-managerial",
                "Prof-specialty",
                "Machine-op-inspct",
                "Adm-clerical",
                "Transport-moving",
                "Protective-serv",
                "Farming-fishing"
            ]
        )

        relationship = st.selectbox(
            "Relationship",
            [
                "Husband",
                "Wife",
                "Not-in-family",
                "Own-child",
                "Unmarried"
            ]
        )

    # -----------------------
    # Right Column
    # -----------------------

    with col2:

        race = st.selectbox(
            "Race",
            [
                "White",
                "Black",
                "Asian-Pac-Islander",
                "Amer-Indian-Eskimo",
                "Other"
            ]
        )

        sex = st.radio(
            "Gender",
            [
                "Male",
                "Female"
            ]
        )

        hours = st.slider(
            "Hours Per Week",
            1,
            100,
            40
        )

        capital_gain = st.number_input(
            "Capital Gain",
            min_value=0,
            value=0,
            step=100
        )

        capital_loss = st.number_input(
            "Capital Loss",
            min_value=0,
            value=0,
            step=100
        )

        country = st.selectbox(
            "Native Country",
            [
                "United-States",
                "India",
                "Canada",
                "England",
                "Mexico"
            ]
        )

    st.divider()

    submitted = st.form_submit_button(
        "🔍 Predict Salary",
        use_container_width=True
    )

    # ==========================================================
# Prediction
# ==========================================================

if submitted:

    input_df = pd.DataFrame({

        "age":[age],
        "workclass":[workclass],
        "education":[education],
        "marital-status":[marital_status],
        "occupation":[occupation],
        "relationship":[relationship],
        "race":[race],
        "sex":[sex],
        "capital-gain":[capital_gain],
        "capital-loss":[capital_loss],
        "hours-per-week":[hours],
        "native-country":[country]

    })

    st.divider()

    st.subheader("📋 Input Summary")

    st.dataframe(
        input_df,
        use_container_width=True,
        hide_index=True
    )

    prediction = pipeline.predict(input_df)[0]

    probability = pipeline.predict_proba(input_df)

    confidence = probability.max() * 100

    st.divider()

    st.subheader("🎯 Prediction Result")

    result_col, confidence_col = st.columns([2,1])

    with result_col:

        if prediction == ">50K":

            st.success(
                "💰 Predicted Income : > $50K"
            )

        else:

            st.error(
                "💰 Predicted Income : ≤ $50K"
            )

    with confidence_col:

        st.metric(
            "Confidence",
            f"{confidence:.2f}%"
        )

        st.progress(confidence/100)

    st.divider()

    st.subheader("📊 Model Information")

    col1,col2,col3,col4 = st.columns(4)

    with col1:

        st.metric(
            "Accuracy",
            f"{metrics['accuracy']*100:.2f}%"
        )

    with col2:

        st.metric(
            "Precision",
            f"{metrics['precision']*100:.2f}%"
        )

    with col3:

        st.metric(
            "Recall",
            f"{metrics['recall']*100:.2f}%"
        )

    with col4:

        st.metric(
            "F1 Score",
            f"{metrics['f1_score']*100:.2f}%"
        )

    st.divider()

    with st.expander("ℹ️ About this Prediction"):

        st.write("""

### Machine Learning Model

This application predicts whether an employee earns more than
**$50,000 per year**.

### Algorithm

Random Forest Classifier

### Dataset

Adult Income Dataset (UCI Repository)

### Technologies

- Python
- Pandas
- NumPy
- Scikit-Learn
- Streamlit
- Joblib

""")

    if "history" not in st.session_state:

        st.session_state.history=[]

    st.session_state.history.insert(

        0,

        {

            "Prediction":prediction,

            "Confidence":f"{confidence:.2f}%"

        }

    )

    with st.expander("🕘 Prediction History"):

        st.dataframe(

            pd.DataFrame(

                st.session_state.history

            ),

            use_container_width=True,

            hide_index=True

        )

# ==========================================================
# Footer
# ==========================================================

st.divider()

st.markdown(

"""
<center>

<h4>Employee Salary Prediction System</h4>

Built using

Python • Pandas • Scikit-Learn • Streamlit

© 2026

</center>

""",

unsafe_allow_html=True

)