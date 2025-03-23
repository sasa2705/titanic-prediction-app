import sqlite3
import pandas as pd
from datetime import datetime

def init_db():
    conn = sqlite3.connect('predictions.db', detect_types=sqlite3.PARSE_DECLTYPES)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            pclass INTEGER NOT NULL,
            sex TEXT NOT NULL,
            age REAL NOT NULL,
            fare REAL NOT NULL,
            embarked TEXT NOT NULL,
            title TEXT NOT NULL,
            family_size INTEGER NOT NULL,
            is_alone INTEGER NOT NULL,
            survival_probability REAL NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

def save_prediction(input_data, survival_prob):
    conn = sqlite3.connect('predictions.db')
    c = conn.cursor()
    
    try:
        print("\nSaving prediction:")
        print("Input data:", input_data)
        print("Survival probability:", survival_prob)
        
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        pclass = int(input_data['Pclass'].iloc[0])
        sex = str(input_data['Sex'].iloc[0])
        age = float(input_data['Age'].iloc[0])
        fare = float(input_data['Fare'].iloc[0])
        embarked = str(input_data['Embarked'].iloc[0])
        title = str(input_data['Title'].iloc[0])
        
        family_size = 1  
        is_alone = 1    
        
        print("\nExtracted values:")
        print(f"pclass: {pclass}, sex: {sex}, age: {age}")
        print(f"fare: {fare}, embarked: {embarked}, title: {title}")
        print(f"family_size: {family_size}, is_alone: {is_alone}")
        
        c.execute('''
            INSERT INTO predictions (
                timestamp, pclass, sex, age, fare, embarked, 
                title, family_size, is_alone, survival_probability
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (timestamp, pclass, sex, age, fare, embarked, title, 
              family_size, is_alone, float(survival_prob)))
        
        conn.commit()
        print("\nSuccessfully saved prediction to database")
        
        c.execute("SELECT * FROM predictions ORDER BY id DESC LIMIT 1")
        last_record = c.fetchone()
        print("\nLast saved record:", last_record)
        
    except Exception as e:
        print(f"\nError saving prediction: {e}")
        print("Input data shape:", input_data.shape)
        print("Input data columns:", input_data.columns)
        print("First row:", input_data.iloc[0].to_dict())
        conn.rollback()
        raise
    finally:
        conn.close()

def get_prediction_history():
    conn = sqlite3.connect('predictions.db')
    try:
        print("\nRetrieving prediction history")
        
        c = conn.cursor()
        c.execute("SELECT COUNT(*) FROM predictions")
        count = c.fetchone()[0]
        print(f"Number of records in database: {count}")
        
        if count == 0:
            print("No records found in database")
            return pd.DataFrame()
        
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
        '''
        
        df = pd.read_sql_query(query, conn)
        print(f"\nRetrieved {len(df)} records")
        
        if not df.empty:
            print("Sample of retrieved data:")
            print(df.head(1))
            return df
        else:
            print("Query returned empty DataFrame")
            return pd.DataFrame()
            
    except Exception as e:
        print(f"\nError retrieving predictions: {e}")
        return pd.DataFrame()
        
    finally:
        conn.close() 
