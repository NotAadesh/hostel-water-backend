import os
from flask import Flask, request, jsonify
import pandas as pd
import psycopg2
from prophet import Prophet

app = Flask(__name__)

DATABASE_URL = os.environ.get("DATABASE_URL")

ALLOWED_AREAS = [
    "HOSTEL 1","HOSTEL 2","HOSTEL 3","HOSTEL 4","HOSTEL 5",
    "HOSTEL 7","HOSTEL 8","HOSTEL 9","HOSTEL 10",
    "CAFETERIA 1","CAFETERIA 2",
    "ACADEMIC BLOCK","NOB",
    "HOUSING FACILITY 1","HOUSING FACILITY 2","HOUSING FACILITY 3",
    "HOUSING FACULTY 4","HOUSING FACULTY 5"
]

def get_connection():
    return psycopg2.connect(DATABASE_URL)

def init_db():
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS readings (
            id SERIAL PRIMARY KEY,
            hostel_name TEXT,
            date DATE,
            meter_reading FLOAT,
            daily_usage FLOAT,
            predicted_usage FLOAT,
            anomaly_flag INTEGER
        );
    """)
    conn.commit()
    cur.close()
    conn.close()

init_db()

def calculate_prediction(area):
    conn = get_connection()
    df = pd.read_sql(
        "SELECT date, daily_usage FROM readings WHERE hostel_name=%s ORDER BY date",
        conn,
        params=(area,)
    )
    conn.close()

    if len(df) < 10:
        return None

    df = df.rename(columns={"date": "ds", "daily_usage": "y"})
    df["ds"] = pd.to_datetime(df["ds"])

    model = Prophet(daily_seasonality=True)
    model.fit(df)

    future = model.make_future_dataframe(periods=1)
    forecast = model.predict(future)

    return float(round(forecast.iloc[-1]["yhat"], 2))

def detect_anomaly(today_usage, predicted):
    if predicted is None:
        return 0
    return 1 if today_usage > predicted * 1.2 else 0

@app.route("/")
def home():
    return "Backend running with PostgreSQL 🚀"

@app.route("/areas")
def areas():
    return jsonify({"areas": ALLOWED_AREAS})

@app.route("/add_reading", methods=["POST"])
def add_reading():
    data = request.json
    area = data["hostel_name"]
    date = data["date"]
    reading = float(data["meter_reading"])

    if area not in ALLOWED_AREAS:
        return jsonify({"error": "Invalid area"}), 400

    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        SELECT meter_reading FROM readings
        WHERE hostel_name=%s
        ORDER BY date DESC LIMIT 1
    """, (area,))
    last = cur.fetchone()

    usage = reading - last[0] if last else 0

    prediction = calculate_prediction(area)
    anomaly = detect_anomaly(usage, prediction)

    cur.execute("""
        INSERT INTO readings
        (hostel_name, date, meter_reading, daily_usage, predicted_usage, anomaly_flag)
        VALUES (%s, %s, %s, %s, %s, %s)
    """, (area, date, reading, usage, prediction, anomaly))

    conn.commit()
    cur.close()
    conn.close()

    return jsonify({
        "daily_usage": usage,
        "predicted_usage": prediction,
        "anomaly_flag": anomaly
    })

@app.route("/dashboard")
def dashboard():
    conn = get_connection()
    df = pd.read_sql("""
        SELECT DISTINCT ON (hostel_name)
        hostel_name, daily_usage, predicted_usage, anomaly_flag
        FROM readings
        ORDER BY hostel_name, date DESC
    """, conn)
    conn.close()

    if df.empty:
        return jsonify({"message": "No data"})

    total_today = float(df["daily_usage"].sum())

    areas_data = df.to_dict(orient="records")

    return jsonify({
        "total_today": total_today,
        "areas": areas_data
    })