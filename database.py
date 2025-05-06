import os
import pandas as pd
import json
from pathlib import Path
import psycopg2
from psycopg2.extras import RealDictCursor
from sqlalchemy import create_engine

def get_db_connection():
    """Get a connection to the PostgreSQL database"""
    db_url = os.environ.get('DATABASE_URL')
    conn = psycopg2.connect(db_url)
    return conn

def get_db_engine():
    """Get SQLAlchemy engine for the PostgreSQL database"""
    db_url = os.environ.get('DATABASE_URL')
    engine = create_engine(db_url)
    return engine

def init_db():
    """Initialize the database with required tables if they don't exist"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Create users table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS users (
        id SERIAL PRIMARY KEY,
        name TEXT NOT NULL,
        email TEXT UNIQUE NOT NULL,
        password TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    
    # Create user_progress table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS user_progress (
        id SERIAL PRIMARY KEY,
        user_id INTEGER NOT NULL,
        module_id INTEGER NOT NULL,
        progress REAL DEFAULT 0,
        completed BOOLEAN DEFAULT FALSE,
        last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users (id)
    )
    ''')
    
    # Create test_results table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS test_results (
        id SERIAL PRIMARY KEY,
        user_id INTEGER NOT NULL,
        module_id INTEGER NOT NULL,
        score REAL NOT NULL,
        completed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users (id)
    )
    ''')
    
    # Check if time_spent_minutes column exists in user_progress and add if it doesn't
    cursor.execute('''
    SELECT column_name FROM information_schema.columns 
    WHERE table_name = 'user_progress' AND column_name = 'time_spent_minutes'
    ''')
    if not cursor.fetchone():
        cursor.execute('''
        ALTER TABLE user_progress ADD COLUMN time_spent_minutes INTEGER DEFAULT 0
        ''')
    
    # Check if visits_count column exists in user_progress and add if it doesn't
    cursor.execute('''
    SELECT column_name FROM information_schema.columns 
    WHERE table_name = 'user_progress' AND column_name = 'visits_count'
    ''')
    if not cursor.fetchone():
        cursor.execute('''
        ALTER TABLE user_progress ADD COLUMN visits_count INTEGER DEFAULT 0
        ''')
    
    conn.commit()
    conn.close()

def get_user_progress(user_id):
    """Get progress for all modules for a specific user"""
    conn = get_db_connection()
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    
    cursor.execute('''
    SELECT module_id, progress, time_spent_minutes, visits_count, 
           last_accessed, completed
    FROM user_progress
    WHERE user_id = %s
    ''', (user_id,))
    
    progress = {}
    progress_details = {}
    
    for row in cursor.fetchall():
        module_id = row['module_id']
        progress[module_id] = row['progress']
        progress_details[module_id] = {
            'progress': row['progress'],
            'time_spent_minutes': row['time_spent_minutes'],
            'visits_count': row['visits_count'],
            'last_accessed': row['last_accessed'],
            'completed': row['completed']
        }
    
    conn.close()
    
    # Store the detailed progress in the session state for access elsewhere
    import streamlit as st
    if 'module_progress_details' not in st.session_state:
        st.session_state.module_progress_details = {}
    st.session_state.module_progress_details = progress_details
    
    return progress

def update_user_progress(user_id, module_id, progress, completed=False, time_spent=0):
    """
    Update a user's progress for a specific module
    
    Parameters:
    - user_id: ID of the user
    - module_id: ID of the module
    - progress: Progress value (0.0 to 1.0)
    - completed: Whether the module is completed
    - time_spent: Additional time spent in minutes (will be added to existing)
    """
    conn = get_db_connection()
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    
    # Check if record exists
    cursor.execute('''
    SELECT id, time_spent_minutes, visits_count FROM user_progress
    WHERE user_id = %s AND module_id = %s
    ''', (user_id, module_id))
    
    record = cursor.fetchone()
    
    if record:
        # Update existing record
        new_time_spent = record['time_spent_minutes'] + time_spent
        new_visits = record['visits_count'] + 1
        
        cursor.execute('''
        UPDATE user_progress
        SET progress = %s, completed = %s, last_accessed = CURRENT_TIMESTAMP,
            time_spent_minutes = %s, visits_count = %s
        WHERE user_id = %s AND module_id = %s
        ''', (progress, completed, new_time_spent, new_visits, user_id, module_id))
    else:
        # Insert new record
        cursor.execute('''
        INSERT INTO user_progress 
        (user_id, module_id, progress, completed, time_spent_minutes, visits_count)
        VALUES (%s, %s, %s, %s, %s, %s)
        ''', (user_id, module_id, progress, completed, time_spent, 1))
    
    conn.commit()
    conn.close()
    
    # Update session state module progress
    import streamlit as st
    if 'module_progress' in st.session_state:
        st.session_state.module_progress[module_id] = progress

def save_test_result(user_id, module_id, score):
    """Save a test result for a user"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute('''
    INSERT INTO test_results (user_id, module_id, score)
    VALUES (%s, %s, %s)
    ''', (user_id, module_id, score))
    
    conn.commit()
    conn.close()

def get_user_test_results(user_id):
    """Get all test results for a specific user"""
    conn = get_db_connection()
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    
    cursor.execute('''
    SELECT module_id, score, completed_at
    FROM test_results
    WHERE user_id = %s
    ORDER BY module_id, completed_at DESC
    ''', (user_id,))
    
    results = []
    for row in cursor.fetchall():
        results.append({
            'module_id': row['module_id'],
            'score': row['score'],
            'completed_at': row['completed_at']
        })
    
    conn.close()
    return results

def create_sample_data():
    """Create sample data files if they don't exist"""
    # Directory for data files
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    # Define sample datasets
    datasets = {
        "fire_incidents.csv": pd.DataFrame({
            "incident_id": range(1, 101),
            "date": pd.date_range(start="2020-01-01", periods=100),
            "detection_time": pd.Series([f"{h:02d}:{m:02d}" for h, m in zip(list(range(24))*5, list(range(0, 60, 6))*4)]),
            "arrival_time": pd.Series([f"{h:02d}:{m:02d}" for h, m in zip(list(range(24))*5, list(range(10, 70, 6))*4)]),
            "extinguishing_time": pd.Series([f"{h:02d}:{m:02d}" for h, m in zip(list(range(24))*5, list(range(30, 90, 6))*4)]),
            "cause": pd.Series(["Electrical"] * 30 + ["Cooking"] * 25 + ["Smoking"] * 20 + ["Heating"] * 15 + ["Other"] * 10),
            "damage_level": pd.Series(["Minor"] * 40 + ["Moderate"] * 35 + ["Severe"] * 25),
        }),
        
        "building_characteristics.csv": pd.DataFrame({
            "building_id": range(1, 51),
            "construction_type": pd.Series(["Wood"] * 15 + ["Concrete"] * 20 + ["Steel"] * 10 + ["Brick"] * 5),
            "age_years": pd.Series([5, 10, 15, 20, 25, 30, 35, 40, 45, 50] * 5),
            "floors": pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 5),
            "has_sprinkler_system": pd.Series([True] * 30 + [False] * 20),
            "has_fire_alarm": pd.Series([True] * 40 + [False] * 10),
            "occupancy_type": pd.Series(["Residential"] * 20 + ["Commercial"] * 15 + ["Industrial"] * 10 + ["Educational"] * 5),
        }),
        
        "sensor_readings.csv": pd.DataFrame({
            "reading_id": range(1, 201),
            "timestamp": pd.date_range(start="2020-01-01", periods=200, freq="1h"),
            "temperature_celsius": pd.Series([20 + i % 60 for i in range(200)]),
            "smoke_density": pd.Series([0.1 + (i % 30) / 10 for i in range(200)]),
            "carbon_monoxide_ppm": pd.Series([5 + i % 100 for i in range(200)]),
            "sensor_id": pd.Series([i % 10 + 1 for i in range(200)]),
            "building_id": pd.Series([i % 50 + 1 for i in range(200)]),
            "alarm_triggered": pd.Series([i % 15 == 0 for i in range(200)]),
        }),
        
        "weather_fire_correlation.csv": pd.DataFrame({
            "date": pd.date_range(start="2020-01-01", periods=100),
            "max_temperature_celsius": pd.Series([20 + i % 30 for i in range(100)]),
            "humidity_percent": pd.Series([30 + i % 50 for i in range(100)]),
            "wind_speed_kmh": pd.Series([5 + i % 50 for i in range(100)]),
            "precipitation_mm": pd.Series([0 + i % 15 for i in range(100)]),
            "fire_incidents_count": pd.Series([1 + i % 10 for i in range(100)]),
        }),
        
        "evacuation_stats.csv": pd.DataFrame({
            "incident_id": range(1, 51),
            "building_id": pd.Series([i % 50 + 1 for i in range(50)]),
            "occupants_total": pd.Series([10 + i * 5 for i in range(50)]),
            "evacuation_time_seconds": pd.Series([60 + i * 10 for i in range(50)]),
            "injuries": pd.Series([i % 5 for i in range(50)]),
            "successful_evacuation_percent": pd.Series([80 + i % 20 for i in range(50)]),
        }),
    }
    
    # Create each dataset file
    for filename, df in datasets.items():
        file_path = data_dir / filename
        if not file_path.exists():
            df.to_csv(file_path, index=False)

# Initialize sample data when module is imported
if __name__ == "__main__":
    init_db()
    create_sample_data()
else:
    # Still create sample data files if they don't exist
    create_sample_data()
