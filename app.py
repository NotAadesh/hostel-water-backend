from flask import Flask, request, jsonify
import sqlite3
import pandas as pd
from prophet import Prophet

app = Flask(__name__)
DB_NAME = "database.db"

# ---------------------
# MASTER AREA LIST
# ---------------------
ALLOWED_AREAS = [
    "HOSTEL 1",
    "HOSTEL 2",
    "HOSTEL 3",
    "HOSTEL 4",
    "HOSTEL 5",
    "HOSTEL 7",
    "HOSTEL 8",
    "HOSTEL 9",
    "HOSTEL 10",
    "CAFETERIA 1",
    "CAFETERIA 2",
    "ACADEMIC BLOCK",
    "NOB",
    "HOUSING FACILITY 1",
    "HOUSING FACILITY 2",
    "HOUSING FACILITY 3",
    "HOUSING FACULTY 4",
    "HOUSING FACULTY 5"
]

# ---------------------
# INIT DATABASE
# ---------------------
def init_db():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS readings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            hostel_name TEXT,
            date TEXT,
            meter_reading REAL,
            daily_usage REAL,
            predicted_usage REAL,
            anomaly_flag INTEGER
        )
    """)
    conn.commit()
    conn.close()

init_db()

# ---------------------
# PROPHET PREDICTION
# ---------------------
def calculate_prediction(hostel):
    conn = sqlite3.connect(DB_NAME)
    df = pd.read_sql_query(
        "SELECT date, daily_usage FROM readings WHERE hostel_name=?",
        conn,
        params=(hostel,)
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

# ---------------------
# ANOMALY DETECTION
# ---------------------
def detect_anomaly(today_usage, predicted):
    if predicted is None:
        return 0
    if today_usage > predicted * 1.2:
        return 1
    return 0

# ---------------------
# ROUTES
# ---------------------

@app.route("/")
def home():
    return "Backend running with Prophet 🚀"

@app.route("/areas")
def get_areas():
    return jsonify({"areas": ALLOWED_AREAS})

@app.route("/add_reading", methods=["POST"])
def add_reading():
    data = request.json
    hostel = data["hostel_name"]
    date = data["date"]
    reading = float(data["meter_reading"])

    if hostel not in ALLOWED_AREAS:
        return jsonify({"error": "Invalid area name"}), 400

    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT meter_reading FROM readings
        WHERE hostel_name=?
        ORDER BY date DESC LIMIT 1
    """, (hostel,))
    
    last = cursor.fetchone()
    usage = reading - last[0] if last else 0

    prediction = calculate_prediction(hostel)
    anomaly = detect_anomaly(usage, prediction)

    cursor.execute("""
        INSERT INTO readings 
        (hostel_name, date, meter_reading, daily_usage, predicted_usage, anomaly_flag)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (hostel, date, reading, usage, prediction, anomaly))

    conn.commit()
    conn.close()

    return jsonify({
        "daily_usage": usage,
        "predicted_usage": prediction,
        "anomaly_flag": anomaly
    })

@app.route("/hostel/<hostel_name>")
def hostel_summary(hostel_name):
    conn = sqlite3.connect(DB_NAME)
    df = pd.read_sql_query(
        "SELECT * FROM readings WHERE hostel_name=? ORDER BY date DESC",
        conn,
        params=(hostel_name,)
    )
    conn.close()

    if df.empty:
        return jsonify({"message": "No data for this area"})

    latest = df.iloc[0]

    return jsonify({
        "area": hostel_name,
        "latest_usage": float(latest["daily_usage"]),
        "predicted_usage": latest["predicted_usage"],
        "anomaly_flag": int(latest["anomaly_flag"])
    })

@app.route("/dashboard")
def dashboard():
    conn = sqlite3.connect(DB_NAME)
    df = pd.read_sql_query("SELECT * FROM readings", conn)
    conn.close()

    if df.empty:
        return jsonify({"message": "No data"})

    latest = df.sort_values("date").groupby("hostel_name").tail(1)

    areas_data = []

    for _, row in latest.iterrows():
        areas_data.append({
            "area": row["hostel_name"],
            "latest_usage": float(row["daily_usage"]),
            "predicted_usage": row["predicted_usage"],
            "anomaly_flag": int(row["anomaly_flag"])
        })

    total_today = float(latest["daily_usage"].sum())

    return jsonify({
        "total_today": total_today,
        "areas": areas_data
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)