# 🚚 Shipment Delay Prediction 🚚

## 📋 Objective
The **Shipment Delay Prediction** project aims to predict the likelihood of shipment delays based on various features like shipment details, weather conditions, and geographical factors. This helps businesses forecast delays and optimize logistics planning, leading to improved customer satisfaction and better operational efficiency. 

## 🛠️ Features
- Model Notebook for Exploratory Data Analysis can find from model1.ipynb file
- API Endpoints: Fast api Predictions 
- Fast api docs
- Feature Importance

## 🧠 Approach
1. **Data Collection**: Gathered historical data on shipments, weather, and traffic patterns.
2. **Data Preprocessing**: Cleaned the data by handling missing values, outliers, and performing feature engineering.
3. **Feature Selection**: Selected the most important features that impact shipment delays.
4. **Model Training**: Used machine learning models like Random Forest,Logistic Regression to predict shipment delays.
5. **Model Evaluation**: Evaluated models using metrics like accuracy, precision, recall, and F1 score.
6. **API**: Used Fast api which serves as inference to accept inputs and return predictions.

## 🤖 Algorithm Used
Employed several machine learning algorithms to predict shipment delays:
- **Random Forest**: Used to handle complex, non-linear relationships in the dataset and provide feature importance insights.
- **Logistic Regression**: A simpler model to establish baseline predictions and compare performance with more complex models.
## How to run
 - After running the shipment_api.py using the command: uvicorn shipment_api:app --reload
 - You can test the api through: http://127.0.0.1:8000/predict url
 - Or you can send request to api through Postman
 - Also, can do using requests library (can find an example in test1.ipynb file)

