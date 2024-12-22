from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Initialize the FastAPI app
app = FastAPI()

# Load the pre-trained model
model = joblib.load('shipment_model.pkl')

# Define label encoders (These should match the encoders used during training)
label_encoders = {
    'Origin': LabelEncoder(),
    'Destination': LabelEncoder(),
    'Vehicle Type': LabelEncoder(),
    'Weather Conditions': LabelEncoder(),
    'Traffic Conditions': LabelEncoder()
}

# Assuming the encoders are fitted on the original dataset during training
# Mock fitting with training categories (replace with actual categories during deployment)
categories = {
    'Origin': ['Jaipur', 'Bangalore', 'Mumbai', 'Hyderabad', 'Chennai', 'Delhi', 'Kolkata', 'Ahmedabad', 'Pune', 'Lucknow'],
    'Destination': ['Mumbai', 'Delhi', 'Chennai', 'Ahmedabad', 'Kolkata','Jaipur', 'Bangalore', 'Hyderabad', 'Pune', 'Lucknow'],
    'Vehicle Type': ['Trailer', 'Truck', 'Container','Lorry'],
    'Weather Conditions': ['Rain', 'Storm', 'Clear', 'Fog'],
    'Traffic Conditions': ['Light', 'Moderate', 'Heavy']
}

for key, values in categories.items():
    label_encoders[key].fit(values)

# Define the input data model
class ShipmentInput(BaseModel):
    Origin: str
    Destination: str
    Vehicle_Type: str
    Distance_km: int
    Weather_Conditions: str
    Traffic_Conditions: str

# Define the prediction endpoint
@app.post("/predict")
def predict_delay(input_data: ShipmentInput):
    # Preprocess the input data
    try:
        encoded_features = [
            label_encoders['Origin'].transform([input_data.Origin])[0],
            label_encoders['Destination'].transform([input_data.Destination])[0],
            label_encoders['Vehicle Type'].transform([input_data.Vehicle_Type])[0],
            input_data.Distance_km,
            label_encoders['Weather Conditions'].transform([input_data.Weather_Conditions])[0],
            label_encoders['Traffic Conditions'].transform([input_data.Traffic_Conditions])[0]
        ]
    except ValueError as e:
        return {"error": f"Invalid input: {e}"}

    # Make prediction
    prediction = model.predict([encoded_features])[0]

    # Map the output to human-readable labels
    result = "Delayed" if prediction == 1 else "On-time"
    
    return {"prediction": result}