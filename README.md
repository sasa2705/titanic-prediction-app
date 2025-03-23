# Titanic Survival Prediction App

A machine learning web application that predicts passenger survival probability on the Titanic using logistic regression. The app includes a prediction history feature with visualizations and statistics.

## Features

- Interactive web interface for making predictions
- Prediction history tracking
- Statistical analysis of predictions
- Data visualizations
- Real-time model training and inference

## Technical Stack

- Streamlit for web interface
- Scikit-learn for machine learning
- SQLite for prediction storage
- Plotly for interactive visualizations
- Pandas for data manipulation

## Setup and Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/YOUR_USERNAME/titanic-survival-prediction.git
   cd titanic-survival-prediction
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the application:
   ```bash
   streamlit run app.py
   ```

## Project Structure

- `app.py`: Main Streamlit application
- `database.py`: Database operations for prediction history
- `requirements.txt`: Project dependencies
- `predictions.db`: SQLite database for storing predictions

## Model Details

The application uses a Logistic Regression model with the following features:
- Passenger Class
- Sex
- Age
- Fare
- Port of Embarkation
- Title (extracted from name)
- Family Size
- Is Alone (derived from family size)

## Deployment

The application is deployed on Streamlit Cloud. Visit [https://share.streamlit.io](https://share.streamlit.io) to access the live version. 
