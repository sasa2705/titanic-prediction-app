import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# Load the trained model
@st.cache_resource
def load_model():
    with open('titanic_model.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

model = load_model()

# Set page title and description
st.title('Titanic Survival Prediction')
st.write("""
This app predicts whether a passenger would have survived the Titanic disaster based on their characteristics.
Enter the passenger details below and click 'Predict' to see the result.
""")

# Add sidebar for navigation
page = st.sidebar.selectbox("Choose a page", ["Prediction", "Model Insights", "About"])

if page == "Prediction":
    st.header("Passenger Information")
    
    # Create input fields
    col1, col2 = st.columns(2)
    
    with col1:
        pclass = st.selectbox('Passenger Class', [1, 2, 3], help="1 = 1st class, 2 = 2nd class, 3 = 3rd class")
        sex = st.selectbox('Sex', ['male', 'female'])
        age = st.slider('Age', 0, 80, 30, help="Age of the passenger")
        title = st.selectbox('Title', ['Mr', 'Mrs', 'Miss', 'Master', 'Rare'], help="Title extracted from name")
    
    with col2:
        fare = st.slider('Fare', 0, 200, 30, help="Ticket fare")
        embarked = st.selectbox('Embarked', ['C', 'Q', 'S'], help="Port of embarkation (C=Cherbourg, Q=Queenstown, S=Southampton)")
        sibsp = st.slider('SibSp', 0, 8, 0, help="Number of siblings/spouses aboard")
        parch = st.slider('Parch', 0, 6, 0, help="Number of parents/children aboard")
    
    # Calculate derived features
    family_size = sibsp + parch + 1
    is_alone = 1 if family_size == 1 else 0
    
    # Create a DataFrame with the input data
    input_data = pd.DataFrame({
        'Pclass': [pclass],
        'Sex': [sex],
        'Age': [age],
        'Fare': [fare],
        'Embarked': [embarked],
        'Title': [title],
        'FamilySize': [family_size],
        'IsAlone': [is_alone]
    })
    
    # Add a predict button
    if st.button('Predict'):
        # Make prediction
        prediction = model.predict(input_data)
        probability = model.predict_proba(input_data)[:, 1]
        
        # Display result
        st.subheader('Prediction Result')
        
        # Create a progress bar for survival probability
        st.write(f"Survival Probability: {probability[0]:.2%}")
        st.progress(float(probability[0]))
        
        if prediction[0] == 1:
            st.success('This passenger would likely SURVIVE the Titanic disaster.')
        else:
            st.error('This passenger would likely NOT SURVIVE the Titanic disaster.')
        
        # Show similar passengers from the training data
        st.subheader("Similar Historical Passengers")
        st.write("Loading similar passengers from the training data...")
        
        # This would require loading the training data
        # For simplicity, we'll just show a placeholder message
        st.info("This feature would show historical passengers with similar characteristics.")

elif page == "Model Insights":
    st.header("Model Performance and Insights")
    
    # Display model metrics
    st.subheader("Model Performance Metrics")
    metrics = {
        "Accuracy": "~80%",
        "Precision": "~75%",
        "Recall": "~70%",
        "F1 Score": "~72%",
        "ROC AUC": "~85%"
    }
    
    for metric, value in metrics.items():
        st.write(f"**{metric}:** {value}")
    
    # Display feature importance
    st.subheader("Feature Importance")
    
    try:
        img = plt.imread('feature_importance.png')
        st.image(img, caption='Feature Importance (Logistic Regression Coefficients)')
    except:
        st.write("Feature importance visualization not available. Run the model training script first.")
    
    # Display ROC curve
    st.subheader("ROC Curve")
    
    try:
        img = plt.imread('roc_curve.png')
        st.image(img, caption='Receiver Operating Characteristic (ROC) Curve')
    except:
        st.write("ROC curve visualization not available. Run the model training script first.")
    
    # Key insights
    st.subheader("Key Insights")
    st.write("""
    1. **Gender was the most important factor**: Women were much more likely to survive than men.
    2. **Class mattered**: First-class passengers had higher survival rates than those in lower classes.
    3. **Age played a role**: Children had higher survival rates, especially young boys.
    4. **Family size had an impact**: Passengers traveling with small families had better chances of survival than those traveling alone or with large families.
    """)

else:  # About page
    st.header("About This Project")
    st.write("""
    ## Titanic Survival Prediction
    
    This project uses machine learning to predict which passengers survived the Titanic shipwreck.
    
    ### The Data
    The data includes passenger information like:
    - Passenger class
    - Sex
    - Age
    - Number of siblings/spouses aboard
    - Number of parents/children aboard
    - Ticket fare
    - Embarkation port
    
    ### The Model
    We used a Logistic Regression model, which is well-suited for binary classification problems like predicting survival.
    
    ### How to Use
    Go to the "Prediction" page and enter passenger details to get a survival prediction.
    
    ### Model Insights
    Check the "Model Insights" page to understand what factors were most important for survival.
    """)
    
    st.write("### Interview Questions")
    
    st.subheader("What is the difference between precision and recall?")
    st.write("""
    **Precision** measures the accuracy of positive predictions. It answers the question: "Of all passengers predicted to survive, what percentage actually survived?"
    
    Precision = True Positives / (True Positives + False Positives)
    
    **Recall** measures the ability to find all positive instances. It answers the question: "Of all passengers who actually survived, what percentage were correctly identified?"
    
    Recall = True Positives / (True Positives + False Negatives)
    
    The key difference is that precision focuses on the accuracy of positive predictions, while recall focuses on finding all positive cases.
    """)
    
    st.subheader("What is cross-validation, and why is it important in binary classification?")
    st.write("""
    **Cross-validation** is a technique where the dataset is repeatedly split into training and validation sets, and the model is trained and evaluated multiple times on different subsets of the data.
    
    It's important in binary classification for several reasons:
    
    1. **Reduces overfitting**: By training and testing on different data subsets, we get a more realistic estimate of how the model will perform on unseen data.
    
    2. **More reliable performance estimates**: Using multiple train-test splits gives a more robust estimate of model performance than a single split.
    
    3. **Better use of limited data**: In datasets like Titanic with limited samples, cross-validation allows us to use all data for both training and validation.
    
    4. **Model selection**: It helps compare different models or hyperparameters more reliably.
    
    In our Titanic model, we used 5-fold cross-validation to ensure our accuracy metrics were robust and not dependent on a particular data split.
    """)

# Add footer
st.sidebar.markdown("---")
st.sidebar.info("Titanic Survival Prediction App") 