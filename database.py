import sqlite3
import os
from datetime import datetime


# ---------------------------------------------------------
# Database path
# ---------------------------------------------------------

PROJECT_PATH = r"C:\Users\HP\OneDrive\Desktop\AI Traffic Accident Risk Prediction"

DATABASE_PATH = os.path.join(
    PROJECT_PATH,
    "predictions.db"
)


# ---------------------------------------------------------
# Create database and table
# ---------------------------------------------------------

def create_database():

    connection = sqlite3.connect(
        DATABASE_PATH
    )

    cursor = connection.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (

            id INTEGER PRIMARY KEY AUTOINCREMENT,

            police_force INTEGER,
            speed_limit INTEGER,

            light_conditions TEXT,
            weather_conditions TEXT,
            road_surface_conditions TEXT,

            special_conditions TEXT,
            carriageway_hazards TEXT,

            urban_or_rural INTEGER,
            year INTEGER,

            predicted_severity INTEGER,

            created_at TEXT
        )
    """)

    connection.commit()
    connection.close()


# ---------------------------------------------------------
# Save prediction
# ---------------------------------------------------------

def save_prediction(
    police_force,
    speed_limit,
    light_conditions,
    weather_conditions,
    road_surface_conditions,
    special_conditions,
    carriageway_hazards,
    urban_or_rural,
    year,
    predicted_severity
):

    connection = sqlite3.connect(
        DATABASE_PATH
    )

    cursor = connection.cursor()

    cursor.execute("""
        INSERT INTO predictions (
            police_force,
            speed_limit,
            light_conditions,
            weather_conditions,
            road_surface_conditions,
            special_conditions,
            carriageway_hazards,
            urban_or_rural,
            year,
            predicted_severity,
            created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        police_force,
        speed_limit,
        light_conditions,
        weather_conditions,
        road_surface_conditions,
        special_conditions,
        carriageway_hazards,
        urban_or_rural,
        year,
        predicted_severity,
        datetime.now().isoformat()
    ))

    connection.commit()
    connection.close()