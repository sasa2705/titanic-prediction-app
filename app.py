import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import tempfile
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from database import init_db, save_prediction, get_prediction_history


init_db()


st.set_page_config(
    page_title="Titanic Survival Prediction",
    page_icon="🚢",
    layout="wide"
)


TEMP_DIR = tempfile.gettempdir()
MODEL_PATH = os.path.join(TEMP_DIR, 'titanic_model.pkl')

@st.cache_data
def load_data():
    
    train_data = pd.read_csv('https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv')
    return train_data

def preprocess_data(df, is_training=True):

    data = df.copy()
    
    data['Title'] = data['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
    
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
    
    data['FamilySize'] = data['SibSp'] + data['Parch'] + 1
    
    data['IsAlone'] = (data['FamilySize'] == 1).astype(int)
    
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
        data = load_data()
        X, y = preprocess_data(data)
        
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

        model = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', LogisticRegression(max_iter=1000, random_state=42))
        ])
        
        model.fit(X, y)
        
        with open(MODEL_PATH, 'wb') as f:
            pickle.dump(model, f)
    
    return model

def display_prediction_history():
    st.header("📊 Prediction History")
    
    history_df = get_prediction_history()
    
    if len(history_df) > 0:
        
        for col in history_df.columns:
            if history_df[col].dtype == object:
                history_df[col] = history_df[col].apply(lambda x: x.decode() if isinstance(x, bytes) else x)
        
        history_df['survival_probability'] = pd.to_numeric(history_df['survival_probability'], errors='coerce')
        history_df['age'] = pd.to_numeric(history_df['age'], errors='coerce')
        history_df['fare'] = pd.to_numeric(history_df['fare'], errors='coerce')
        
        history_df['survival_probability'] = history_df['survival_probability'].fillna(0)
        history_df['age'] = history_df['age'].fillna(0)
        history_df['fare'] = history_df['fare'].fillna(0)
        
        tab1, tab2, tab3 = st.tabs(["Recent Predictions", "Statistics", "Visualizations"])
        
        with tab1:
            st.dataframe(
                history_df.style.format({
                    'survival_probability': '{:.1%}',
                    'age': '{:.0f}',
                    'fare': '${:.2f}'
                }),
                hide_index=True
            )
        
        with tab2:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                avg_survival = history_df['survival_probability'].mean()
                st.metric("Average Survival Probability", f"{avg_survival:.1%}")
            
            with col2:
                total_predictions = len(history_df)
                st.metric("Total Predictions", total_predictions)
            
            with col3:
                high_survival = (history_df['survival_probability'] >= 0.7).mean()
                st.metric("High Survival Rate", f"{high_survival:.1%}")
        
        with tab3:
            fig1 = px.histogram(
                history_df,
                x='survival_probability',
                title='Distribution of Survival Probabilities',
                labels={'survival_probability': 'Survival Probability'},
                nbins=20
            )
            st.plotly_chart(fig1)
            
            avg_by_class_sex = history_df.groupby(['pclass', 'sex'])['survival_probability'].mean().reset_index()
            fig2 = px.bar(
                avg_by_class_sex,
                x='pclass',
                y='survival_probability',
                color='sex',
                title='Average Survival Probability by Class and Sex',
                labels={
                    'pclass': 'Passenger Class',
                    'survival_probability': 'Average Survival Probability',
                    'sex': 'Sex'
                }
            )
            st.plotly_chart(fig2)
    else:
        st.info("No predictions have been made yet. Make some predictions to see the history!")

def main():
    st.title("🚢 Titanic Survival Prediction")
    
    tab1, tab2 = st.tabs(["Make Prediction", "View History"])
    
    with tab1:
        st.write("Enter passenger information to predict survival probability")
        
        model = load_or_train_model()
        
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
            
            survival_prob = model.predict_proba(input_data)[0][1]
            
            save_prediction(input_data, survival_prob)
            
            st.header("Prediction Result")
            
            st.progress(survival_prob)
            
            st.metric("Survival Probability", f"{survival_prob:.1%}")
            
            st.write("### Interpretation")
            if survival_prob >= 0.7:
                st.success("High chance of survival! 🎉")
            elif survival_prob >= 0.3:
                st.warning("Moderate chance of survival 😐")
            else:
                st.error("Low chance of survival 😢")
    
    with tab2:
        display_prediction_history()

if __name__ == "__main__":
    main()
