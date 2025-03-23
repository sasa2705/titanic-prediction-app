import sqlite3
import pandas as pd
from datetime import datetime

def init_db():
    conn = sqlite3.connect('predictions.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            pclass INTEGER,
            sex TEXT,
            age REAL,
            fare REAL,
            embarked TEXT,
            title TEXT,
            family_size INTEGER,
            is_alone INTEGER,
            survival_probability REAL
        )
    ''')
    conn.commit()
    conn.close()

def save_prediction(input_data, survival_prob):
    conn = sqlite3.connect('predictions.db')
    c = conn.cursor()
    
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    c.execute('''
        INSERT INTO predictions (
            timestamp, pclass, sex, age, fare, embarked, 
            title, family_size, is_alone, survival_probability
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        timestamp,
        input_data['Pclass'].iloc[0],
        input_data['Sex'].iloc[0],
        input_data['Age'].iloc[0],
        input_data['Fare'].iloc[0],
        input_data['Embarked'].iloc[0],
        input_data['Title'].iloc[0],
        input_data['FamilySize'].iloc[0],
        input_data['IsAlone'].iloc[0],
        survival_prob
    ))
    
    conn.commit()
    conn.close()

def get_prediction_history():
    conn = sqlite3.connect('predictions.db')
    try:
        df = pd.read_sql_query('''
            SELECT 
                timestamp,
                pclass,
                sex,
                age,
                fare,
                embarked,
                title,
                family_size,
                is_alone,
                survival_probability
            FROM predictions 
            ORDER BY timestamp DESC
        ''', conn)
        
        # Ensure proper string encoding for text columns
        text_columns = ['timestamp', 'sex', 'embarked', 'title']
        for col in text_columns:
            df[col] = df[col].astype(str)
            
        # Convert numeric columns
        df['pclass'] = pd.to_numeric(df['pclass'], errors='coerce')
        df['age'] = pd.to_numeric(df['age'], errors='coerce')
        df['fare'] = pd.to_numeric(df['fare'], errors='coerce')
        df['family_size'] = pd.to_numeric(df['family_size'], errors='coerce')
        df['is_alone'] = pd.to_numeric(df['is_alone'], errors='coerce')
        df['survival_probability'] = pd.to_numeric(df['survival_probability'], errors='coerce')
        
    except Exception as e:
        print(f"Error reading from database: {e}")
        df = pd.DataFrame()  # Return empty DataFrame on error
    finally:
        conn.close()
    
    return df 
