import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import tempfile
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

# Set page config
st.set_page_config(page_title="Titanic Survival Prediction", page_icon="🚢")

# Create a temporary directory for the model file
TEMP_DIR = tempfile.gettempdir()
MODEL_PATH = os.path.join(TEMP_DIR, 'titanic_model.pkl')

@st.cache_data
def load_data():
    # Load sample data for demonstration
    train_data = pd.read_csv('https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv')
    return train_data

def preprocess_data(df, is_training=True):
    # Create a copy to avoid modifying the original
    data = df.copy()
    
    # Extract titles from names
    data['Title'] = data['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
    
    # Group rare titles
    title_mapping = {
        'Mr': 'Mr',
        'Miss': 'Miss',
        'Mrs': 'Mrs',
        'Master': 'Master',
        'Dr': 'Rare',
        'Rev': 'Rare',
        'Col': 'Rare',
        'Major': 'Rare',
        'Mlle': 'Miss',
        'Countess': 'Rare',
        'Ms': 'Miss',
        'Lady': 'Rare',
        'Jonkheer': 'Rare',
        'Don': 'Rare',
        'Dona': 'Rare',
        'Mme': 'Mrs',
        'Capt': 'Rare',
        'Sir': 'Rare'
    }
    data['Title'] = data['Title'].map(lambda x: title_mapping.get(x, 'Rare'))
    
    # Create family size feature
    data['FamilySize'] = data['SibSp'] + data['Parch'] + 1
    
    # Create is_alone feature
    data['IsAlone'] = (data['FamilySize'] == 1).astype(int)
    
    # Select features for the model
    features = ['Pclass', 'Sex', 'Age', 'Fare', 'Embarked', 'Title', 'FamilySize', 'IsAlone']
    
    if is_training:
        X = data[features]
        y = data['Survived']
        return X, y
    else:
        return data[features]

@st.cache_resource
def load_or_train_model():
    try:
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
    except FileNotFoundError:
        # Load and preprocess data
        data = load_data()
        X, y = preprocess_data(data)
        
        # Define preprocessing pipeline
        numeric_features = ['Age', 'Fare', 'FamilySize']
        categorical_features = ['Pclass', 'Sex', 'Embarked', 'Title', 'IsAlone']

        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ])

        # Create and train the model
        model = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', LogisticRegression(max_iter=1000, random_state=42))
        ])
        
        # Train the model
        model.fit(X, y)
        
        # Save the model to temporary directory
        with open(MODEL_PATH, 'wb') as f:
            pickle.dump(model, f)
    
    return model

# Main app
def main():
    st.title("🚢 Titanic Survival Prediction")
    st.write("Enter passenger information to predict survival probability")
    
    # Load or train the model
    model = load_or_train_model()
    
    # Create input form
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            pclass = st.selectbox("Passenger Class", options=[1, 2, 3], help="1 = 1st class, 2 = 2nd class, 3 = 3rd class")
            sex = st.selectbox("Sex", options=['male', 'female'])
            age = st.number_input("Age", min_value=0, max_value=100, value=30)
            title = st.selectbox("Title", options=['Mr', 'Mrs', 'Miss', 'Master', 'Rare'])
        
        with col2:
            fare = st.number_input("Fare", min_value=0, max_value=600, value=32)
            embarked = st.selectbox("Embarked", options=['C', 'Q', 'S'], help="C = Cherbourg, Q = Queenstown, S = Southampton")
            siblings = st.number_input("Number of Siblings/Spouses Aboard", min_value=0, max_value=10, value=0)
            parents = st.number_input("Number of Parents/Children Aboard", min_value=0, max_value=10, value=0)
        
        submit_button = st.form_submit_button("Predict Survival")
    
    if submit_button:
        # Create a DataFrame with user input
        input_data = pd.DataFrame({
            'Pclass': [pclass],
            'Sex': [sex],
            'Age': [age],
            'Fare': [fare],
            'Embarked': [embarked],
            'Title': [title],
            'FamilySize': [siblings + parents + 1],
            'IsAlone': [1 if siblings + parents == 0 else 0]
        })
        
        # Make prediction
        survival_prob = model.predict_proba(input_data)[0][1]
        
        # Display result
        st.header("Prediction Result")
        
        # Create a progress bar for survival probability
        st.progress(survival_prob)
        
        # Display probability as percentage
        st.metric("Survival Probability", f"{survival_prob:.1%}")
        
        # Interpretation
        st.write("### Interpretation")
        if survival_prob >= 0.7:
            st.success("High chance of survival! 🎉")
        elif survival_prob >= 0.3:
            st.warning("Moderate chance of survival 😐")
        else:
            st.error("Low chance of survival 😢")

if __name__ == "__main__":
    main()
