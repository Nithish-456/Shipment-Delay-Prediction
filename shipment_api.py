from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

# Load the trained model
model = joblib.load('shipment_model.pkl')

# Create the FastAPI app
app = FastAPI()

# Define the request body model
class ShipmentDetails(BaseModel):
    Origin: str
    Destination: str
    Vehicle_Type: str
    Distance_km: float
    Weather_Conditions: str
    Traffic_Conditions: str

@app.post("/predict/")
async def predict_shipment_delay(details: ShipmentDetails):
    # Convert the input data to a DataFrame
    input_data = pd.DataFrame([{
        'Origin': details.Origin,
        'Destination': details.Destination,
        'Vehicle Type': details.Vehicle_Type,
        'Distance (km)': details.Distance_km,
        'Weather Conditions': details.Weather_Conditions,
        'Traffic Conditions': details.Traffic_Conditions
    }])

    # Make the prediction
    prediction = model.predict(input_data)

    # Return the prediction result
    return {"prediction": "Delayed" if prediction[0] == 1 else "On Time"}