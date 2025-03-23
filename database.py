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
    
    try:
        pclass = int(input_data.loc[0, 'Pclass'])
        age = float(input_data.loc[0, 'Age'])
        fare = float(input_data.loc[0, 'Fare'])
        
        values = (
            timestamp,
            pclass,
            str(input_data.loc[0, 'Sex']),
            age,
            fare,
            str(input_data.loc[0, 'Embarked']),
            str(input_data.loc[0, 'Title']),
            int(input_data.loc[0, 'FamilySize']),
            int(input_data.loc[0, 'IsAlone']),
            float(survival_prob)
        )
        
        c.execute('''
            INSERT INTO predictions (
                timestamp, pclass, sex, age, fare, embarked, 
                title, family_size, is_alone, survival_probability
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', values)
        
        conn.commit()
        print(f"Saved values: pclass={pclass}, age={age}, fare={fare}")  
        
    except Exception as e:
        print(f"Error saving prediction: {e}")
        conn.rollback()
    finally:
        conn.close()

def get_prediction_history():
    conn = sqlite3.connect('predictions.db')
    try:
        df = pd.read_sql_query('''
            SELECT 
                timestamp,
                CAST(pclass AS INTEGER) as pclass,
                sex,
                CAST(age AS FLOAT) as age,
                CAST(fare AS FLOAT) as fare,
                embarked,
                title,
                family_size,
                is_alone,
                survival_probability
            FROM predictions 
            ORDER BY timestamp DESC
        ''', conn)
        
        text_columns = ['timestamp', 'sex', 'embarked', 'title']
        for col in text_columns:
            df[col] = df[col].astype(str)
        
        df['pclass'] = df['pclass'].astype(int)
        df['age'] = df['age'].astype(float)
        df['fare'] = df['fare'].astype(float)
        df['family_size'] = df['family_size'].astype(int)
        df['is_alone'] = df['is_alone'].astype(int)
        df['survival_probability'] = df['survival_probability'].astype(float)
        
    except Exception as e:
        print(f"Error reading from database: {e}")
        df = pd.DataFrame()
    finally:
        conn.close()
    
    return df 
