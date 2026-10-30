import pandas as pd
import streamlit as st

from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

st.markdown("""
<style>

/* Main background */
.stApp {
    background-color:  #F5E6D3;
}

/* Main title */
h1 {
    color: #1e3a5f;
    font-size: 38px;
    font-weight: bold;
    text-align: center;
}

/* Subheadings */
h2, h3 {
    color: #24527a;
}

/* Input labels */
label {
    color: #263746 !important;
    font-weight: 600 !important;
}

/* Input boxes */
.stSelectbox, .stNumberInput {
    background-color: transparent !important;
    border-radius: 8px;
}

/* Predict button */
div.stButton > button {
    background-color: #1565c0;
    color: white;
    border-radius: 8px;
    border: none;
    padding: 10px 25px;
    font-weight: bold;
    width: 100%;
}

div.stButton > button:hover {
    background-color: #F3E5D0;
    color: white;
}

</style>
""", unsafe_allow_html=True)

st.markdown("""
    <h1 style="
        white-space: nowrap;
        font-size: clamp(20px, 3.5vw, 42px);
        text-align: center;
        margin: 0;
    ">
        🚨 TRAFFIC ACCIDENT IMPACT PREDICTION
    </h1>
""", unsafe_allow_html=True)

# st.title("🚨TRAFFIC ACCIDENT SEVERITY PREDICTION")

data=pd.read_csv(r"C:\Users\Gosukula Shirisha\OneDrive\Desktop\ML Supervised Project\MLreport")

data['Accident_Severity']=data['Accident_Severity'].map({'Minor':0,'Moderate':1,'Severe':2})

x=data[['Accident_Hour','Area_Type','Weather','Road_Condition','Vehicle_Type','Speed_Limit_kmh','Number_of_Vehicles','Traffic_Volume']].copy()

x['Area_Type']=x['Area_Type'].map({'Urban': 1, 'Suburban': 2,'Rural': 3,'Unknown': 0})
x['Weather']=x['Weather'].map({ 'Clear': 0,'Rainy': 1,'Foggy': 2,'Snowy': 3,'Unknown': 4})
x['Road_Condition']=x['Road_Condition'].map({'Dry': 0,'Wet': 1,'Icy': 2, 'Unknown': 3})
x['Vehicle_Type']=x['Vehicle_Type'].map({'Car': 0,'Bike': 1, 'Bus': 2,'Truck': 3,'Unknown': 4})

y=data['Accident_Severity']

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.30)
model=XGBClassifier(n_estimators=300,random_state=42,max_depth=3,learning_rate=0.05)

model.fit(x_train,y_train)

# st.write("Number of features:", model.n_features_in_)

st.subheader("📝Enter Accident Details")

accident_hour=st.number_input("Enter accident hour (0-23)",min_value=0, max_value=23, step=1, key="accident_hour")
area=st.number_input("Enter Area Type: Urban=1, Suburban=2, Rural=3, Unknown=0",min_value=0, max_value=3, step=1, key="area")
weather = st.number_input("Enter Weather: Clear=0, Rainy=1, Foggy=2, Snowy=3, Unknown=4", min_value=0, max_value=4, step=1, key="weather")
road = st.number_input("Enter Road Condition: Dry=0, Wet=1, Icy=2, Unknown=3", min_value=0, max_value=3, step=1,key="road")
vehicle = st.number_input("Enter Vehicle Type: Car=0, Bike=1, Bus=2, Truck=3, Unknown=4",min_value=0, max_value=4, step=1,key="vehicle")
speed = st.number_input("Enter Speed Limit (km/h)", min_value=1, step=1,key="speed")
vehicles = st.number_input("Enter Number of Vehicles",min_value=1, step=1,key="vehicles")
traffic = st.number_input( "Enter Traffic Volume",min_value=0, step=1,key="traffic")

button = st.button("Predict Accident Severity")

if button:
    input_data=[[ accident_hour,area, weather,road,vehicle,speed, vehicles,traffic]]
    
    output=model.predict(input_data)
    
    severity={
        0: "Minor",
        1: "Moderate",
        2: "Severe"
    }
    result = severity[int(output[0])]
    
    if result == "Minor":
        st.success("Predicted Accident Severity: Minor")
    elif result == "Moderate":
        st.warning("Predicted Accident Severity: Moderate")
    else:
        st.error("Predicted Accident Severity: Severe")