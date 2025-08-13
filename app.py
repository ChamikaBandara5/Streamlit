import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

# ------------------ PAGE CONFIG ------------------
st.set_page_config(
    page_title="🚢 Titanic Survival Predictor",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------ CUSTOM CSS & BACKGROUND ------------------
def set_background_color():
    st.markdown(
         f"""
         <style>
         /* Background Color */
         .stApp {{
             background-color: #1E1E1E;  /* Dark theme */
             font-family: 'Poppins', sans-serif;
         }}

         /* Headings Style */
         h1, h2, h3, h4 {{
             color: #FFD700 !important;
             text-shadow: 2px 2px 5px rgba(0,0,0,0.6);
         }}

         /* Metrics Box */
         [data-testid="stMetricValue"] {{
             color: #FFD700 !important;
         }}

         /* Sidebar Styling */
         section[data-testid="stSidebar"] {{
             background-color: #2E2E2E;
             color: white;
         }}

         /* DataFrame Table */
         .dataframe tbody tr {{
             background-color: rgba(255,255,255,0.95);
         }}

         /* Prediction Box */
         .prediction-result {{
             background-color: rgba(0, 0, 0, 0.7);
             color: white;
             padding: 15px;
             border-radius: 10px;
             text-align: center;
             font-size: 18px;
             margin-top: 20px;
         }}

         /* Custom Button */
         div.stButton > button:first-child {{
             background-color: #ff4b4b;
             color: white;
             border-radius: 8px;
             font-size: 18px;
             padding: 10px 24px;
             transition: 0.3s;
         }}
         div.stButton > button:first-child:hover {{
             background-color: #ff0000;
             transform: scale(1.05);
         }}
         </style>
         """,
         unsafe_allow_html=True
     )

set_background_color()


# ------------------ LOAD DATA ------------------
@st.cache_data
def load_data():
    df = pd.read_csv("Titanic-Dataset.csv")
    return df

@st.cache_resource
def load_model():
    with open("best_model.pkl", "rb") as file:
        model = pickle.load(file)
    return model

df = load_data()
model = load_model()

# ------------------ SIDEBAR NAV ------------------
st.sidebar.title("📌 Navigation")
page = st.sidebar.radio("Go to", ["🏠 Home", "🔍 Data Exploration", "📊 Visualisation", "🤖 Make Prediction", "📈 Model Performance"])

# ------------------ HOME PAGE ------------------
if page == "🏠 Home":
    st.markdown(
        """
        <div style="text-align: center;">
            <h1>🚢 Titanic Survival Prediction App</h1>
            <h3>Predict the survival chances of Titanic passengers based on their details</h3>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Titanic Image
    st.image("https://upload.wikimedia.org/wikipedia/commons/f/fd/RMS_Titanic_3.jpg", use_column_width=True)

    st.markdown(
        """
        <p style="color: white; font-size: 18px; text-align: center;">
        This app uses machine learning to predict whether a passenger on the Titanic would survive, 
        based on various features such as age, gender, passenger class, and more.
        </p>
        """,
        unsafe_allow_html=True
    )

# ------------------ DATA EXPLORATION ------------------
elif page == "🔍 Data Exploration":
    st.title("🔍 Titanic Dataset Exploration")
    st.write("Explore the dataset: shape, columns, datatypes, sample data.")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Passengers", df.shape[0])
    col2.metric("Features", df.shape[1])
    col3.metric("Missing Values", df.isnull().sum().sum())

    st.write("### Dataset Preview")
    st.dataframe(df.head())

    st.write("### Column Information")
    st.write(df.dtypes)

    st.write("### Interactive Filtering")
    sex_filter = st.selectbox("Filter by Sex", options=["All"] + df["Sex"].unique().tolist())
    pclass_filter = st.selectbox("Filter by Pclass", options=["All"] + df["Pclass"].astype(str).unique().tolist())

    filtered_df = df.copy()
    if sex_filter != "All":
        filtered_df = filtered_df[filtered_df["Sex"] == sex_filter]
    if pclass_filter != "All":
        filtered_df = filtered_df[filtered_df["Pclass"].astype(str) == pclass_filter]

    st.dataframe(filtered_df)

# ------------------ VISUALISATION ------------------
elif page == "📊 Visualisation":
    st.title("📊 Titanic Data Visualisation")

    st.subheader("Survival by Gender")
    st.plotly_chart(px.histogram(df, x="Sex", color="Survived", barmode="group"), use_container_width=True)

    st.subheader("Survival by Passenger Class")
    st.plotly_chart(px.histogram(df, x="Pclass", color="Survived", barmode="group"), use_container_width=True)

    st.subheader("Age Distribution")
    st.plotly_chart(px.histogram(df, x="Age", nbins=30, color="Survived"), use_container_width=True)

# ------------------ PREDICTION ------------------
elif page == "🤖 Make Prediction":
    st.title("🤖 Titanic Survival Prediction")
    st.write("Enter passenger details below:")

    col1, col2 = st.columns(2)
    with col1:
        pclass = st.selectbox("Passenger Class (1 = 1st, 2 = 2nd, 3 = 3rd)", [1, 2, 3])
        sex = st.selectbox("Sex", ["male", "female"])
        age = st.number_input("Age", min_value=0, max_value=100, value=30)
        sibsp = st.number_input("Number of Siblings/Spouses Aboard", min_value=0, max_value=10, value=0)

    with col2:
        parch = st.number_input("Number of Parents/Children Aboard", min_value=0, max_value=10, value=0)
        fare = st.number_input("Passenger Fare", min_value=0.0, max_value=600.0, value=32.0)
        embarked = st.selectbox("Port of Embarkation", ["C", "Q", "S"])

    if st.button("🚢 Predict Survival", type="primary"):
        try:
            input_df = pd.DataFrame([{
                "Pclass": pclass,
                "Sex": sex,
                "Age": age,
                "SibSp": sibsp,
                "Parch": parch,
                "Fare": fare,
                "Embarked": embarked
            }])
            input_df["FamilySize"] = input_df["SibSp"] + input_df["Parch"] + 1
            input_df["FarePerPerson"] = input_df["Fare"] / input_df["FamilySize"]
            input_df["IsAlone"] = (input_df["FamilySize"] == 1).astype(int)
            input_df = pd.get_dummies(input_df, columns=["Sex", "Embarked"], drop_first=False)
            input_df = input_df.reindex(columns=model.feature_names_in_, fill_value=0)

            prediction = model.predict(input_df)[0]
            probability = model.predict_proba(input_df)[0][prediction]

            result = "Survived" if prediction == 1 else "Did Not Survive"
            color = "green" if prediction == 1 else "red"

            st.markdown(f"<div class='prediction-result'><h2 style='color:{color}'>{result}</h2><p>Confidence: {probability:.2%}</p></div>", unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Error making prediction: {e}")

# ------------------ MODEL PERFORMANCE ------------------
elif page == "📈 Model Performance":
    st.title("📈 Model Performance")
    data = df.dropna(subset=["Age", "Embarked", "Fare", "Sex", "Pclass", "SibSp", "Parch"])
    data["FamilySize"] = data["SibSp"] + data["Parch"] + 1
    data["FarePerPerson"] = data["Fare"] / data["FamilySize"]
    data["IsAlone"] = (data["FamilySize"] == 1).astype(int)
    data = pd.get_dummies(data, columns=["Sex", "Embarked"], drop_first=False)
    data = data.reindex(columns=model.feature_names_in_, fill_value=0)

    X = data
    y = df.loc[data.index, "Survived"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    y_pred = model.predict(X_test)

    st.write(f"**Accuracy:** {accuracy_score(y_test, y_pred):.3f}")
    st.write(f"**Precision:** {precision_score(y_test, y_pred):.3f}")
    st.write(f"**Recall:** {recall_score(y_test, y_pred):.3f}")
    st.write(f"**F1 Score:** {f1_score(y_test, y_pred):.3f}")

    st.subheader("Classification Report")
    st.text(classification_report(y_test, y_pred))

    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues",
                xticklabels=["Did Not Survive", "Survived"],
                yticklabels=["Did Not Survive", "Survived"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)
