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
        # Access first row values using dictionary-style access
        values = (
            timestamp,
            int(input_data['Pclass'].iloc[0]),
            str(input_data['Sex'].iloc[0]),
            float(input_data['Age'].iloc[0]),
            float(input_data['Fare'].iloc[0]),
            str(input_data['Embarked'].iloc[0]),
            str(input_data['Title'].iloc[0]),
            int(input_data['FamilySize'].iloc[0]),
            int(input_data['IsAlone'].iloc[0]),
            float(survival_prob)
        )
        
        # Print debug info before saving
        print("Attempting to save:", values)
        
        c.execute('''
            INSERT INTO predictions 
            (timestamp, pclass, sex, age, fare, embarked, title, family_size, is_alone, survival_probability)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', values)
        
        conn.commit()
        print("Successfully saved prediction to database")
        
    except Exception as e:
        print(f"Error saving prediction: {e}")
        print("Input data:", input_data)
        conn.rollback()
    finally:
        conn.close()

def get_prediction_history():
    conn = sqlite3.connect('predictions.db')
    try:
        # Simplified query with explicit column selection
        query = '''
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
        LIMIT 100
        '''
        
        df = pd.read_sql_query(query, conn)
        
        if len(df) > 0:
            # Convert types safely
            numeric_cols = {
                'pclass': 'int32',
                'age': 'float64',
                'fare': 'float64',
                'family_size': 'int32',
                'is_alone': 'int32',
                'survival_probability': 'float64'
            }
            
            for col, dtype in numeric_cols.items():
                try:
                    df[col] = pd.to_numeric(df[col], errors='coerce').astype(dtype)
                except Exception as e:
                    print(f"Error converting {col}: {e}")
            
            # Convert string columns
            string_cols = ['timestamp', 'sex', 'embarked', 'title']
            for col in string_cols:
                df[col] = df[col].fillna('').astype(str)
        
        print(f"Retrieved {len(df)} records from database")
        print("Sample data:", df.head(1).to_dict('records'))
        
    except Exception as e:
        print(f"Error reading from database: {e}")
        df = pd.DataFrame()
    finally:
        conn.close()
    
    return df 
