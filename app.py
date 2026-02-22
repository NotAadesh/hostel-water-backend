from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import psycopg2
import os
import io
import pandas as pd
from prophet import Prophet
from datetime import timedelta

app = Flask(__name__)
CORS(app)


# ----------------------------------
# DATABASE CONNECTION
# ----------------------------------
def get_connection():
    url = os.environ.get("DATABASE_URL")

    if not url:
        raise Exception("DATABASE_URL not set")

    if url.startswith("postgres://"):
        url = url.replace("postgres://", "postgresql://", 1)

    return psycopg2.connect(url)


# ----------------------------------
# INIT DATABASE (MANUAL)
# ----------------------------------
def init_db():
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("DROP TABLE IF EXISTS readings;")

    cur.execute("""
        CREATE TABLE readings (
            id SERIAL PRIMARY KEY,
            hostel_name VARCHAR(100),
            date DATE,
            domestic_reading FLOAT,
            flush_reading FLOAT,
            domestic_usage FLOAT,
            flush_usage FLOAT,
            total_usage FLOAT,
            anomaly_flag INT DEFAULT 0
        );
    """)

    conn.commit()
    cur.close()
    conn.close()


@app.route("/reset_db")
def reset_db():
    init_db()
    return {"message": "Database reset complete"}


# ----------------------------------
# AREAS
# ----------------------------------
AREAS = [
    "HOSTEL 1", "HOSTEL 2", "HOSTEL 3", "HOSTEL 4",
    "HOSTEL 5", "HOSTEL 6", "HOSTEL 7", "HOSTEL 8",
    "HOSTEL 9", "HOSTEL 10",
    "CAFETERIA 1", "CAFETERIA 2",
    "ACADEMIC BLOCK",
    "NOB",
    "HOUSING FACILITY 1", "HOUSING FACILITY 2",
    "HOUSING FACILITY 3", "HOUSING FACILITY 4",
    "HOUSING FACILITY 5"
]


@app.route("/areas", methods=["GET"])
def get_areas():
    return jsonify({"areas": AREAS})


# ----------------------------------
# ADD READING
# ----------------------------------
@app.route("/add_reading", methods=["POST"])
def add_reading():
    data = request.get_json()

    hostel = data["hostel_name"]
    date_val = data["date"]
    domestic_today = float(data["domestic_reading"])
    flush_today = float(data["flush_reading"])

    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        SELECT domestic_reading, flush_reading
        FROM readings
        WHERE hostel_name = %s
        ORDER BY date DESC
        LIMIT 1
    """, (hostel,))

    prev = cur.fetchone()

    if prev:
        prev_domestic = prev[0]
        prev_flush = prev[1]
    else:
        prev_domestic = 0
        prev_flush = 0

    domestic_usage = domestic_today - prev_domestic
    flush_usage = flush_today - prev_flush
    total_usage = domestic_usage + flush_usage

    anomaly = 1 if total_usage > 500 else 0

    cur.execute("""
        INSERT INTO readings (
            hostel_name, date,
            domestic_reading, flush_reading,
            domestic_usage, flush_usage, total_usage,
            anomaly_flag
        )
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
    """, (
        hostel, date_val,
        domestic_today, flush_today,
        domestic_usage, flush_usage, total_usage,
        anomaly
    ))

    conn.commit()
    cur.close()
    conn.close()

    return jsonify({
        "domestic_usage": domestic_usage,
        "flush_usage": flush_usage,
        "total_usage": total_usage,
        "anomaly_flag": anomaly
    })


# ----------------------------------
# DASHBOARD
# ----------------------------------
@app.route("/dashboard", methods=["GET"])
def dashboard():

    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        SELECT hostel_name,
               domestic_usage,
               flush_usage,
               total_usage,
               anomaly_flag
        FROM readings
        WHERE date = (SELECT MAX(date) FROM readings)
    """)

    rows = cur.fetchall()

    areas_data = []
    total_today = 0
    total_domestic = 0
    total_flush = 0

    for row in rows:
        areas_data.append({
            "hostel_name": row[0],
            "domestic_usage": row[1],
            "flush_usage": row[2],
            "total_usage": row[3],
            "anomaly_flag": row[4]
        })

        total_domestic += row[1]
        total_flush += row[2]
        total_today += row[3]

    top_areas = sorted(
        areas_data,
        key=lambda x: x["total_usage"],
        reverse=True
    )[:3]

    cur.close()
    conn.close()

    return jsonify({
        "areas": areas_data,
        "total_today": total_today,
        "total_domestic": total_domestic,
        "total_flush": total_flush,
        "top_areas": top_areas
    })


# ----------------------------------
# TREND + PREDICTION (3 DAYS)
# ----------------------------------
@app.route("/trend", methods=["GET"])
def trend():

    selected = request.args.get("areas")

    if not selected:
        selected_areas = AREAS
    else:
        selected_areas = selected.split(",")

    conn = get_connection()

    df = pd.read_sql("""
        SELECT date,
               hostel_name,
               domestic_usage,
               flush_usage
        FROM readings
        ORDER BY date ASC
    """, conn)

    conn.close()

    if df.empty:
        return {"error": "No data available"}, 404

    df = df[df["hostel_name"].isin(selected_areas)]

    grouped = df.groupby("date").agg({
        "domestic_usage": "sum",
        "flush_usage": "sum"
    }).reset_index()

    # Prophet requires ds and y columns
    domestic_df = grouped[["date", "domestic_usage"]].rename(
        columns={"date": "ds", "domestic_usage": "y"}
    )

    flush_df = grouped[["date", "flush_usage"]].rename(
        columns={"date": "ds", "flush_usage": "y"}
    )

    # Train Prophet models
    model_domestic = Prophet()
    model_domestic.fit(domestic_df)

    model_flush = Prophet()
    model_flush.fit(flush_df)

    future_domestic = model_domestic.make_future_dataframe(periods=3)
    forecast_domestic = model_domestic.predict(future_domestic)

    future_flush = model_flush.make_future_dataframe(periods=3)
    forecast_flush = model_flush.predict(future_flush)

    result = []

    for i in range(len(forecast_domestic)):
        result.append({
            "date": forecast_domestic["ds"].iloc[i].strftime("%Y-%m-%d"),
            "domestic": float(forecast_domestic["yhat"].iloc[i]),
            "flush": float(forecast_flush["yhat"].iloc[i]),
            "is_forecast": i >= len(grouped)
        })

    return jsonify({
        "data": result
    })


@app.route("/")
def home():
    return {"message": "Full Water Intelligence Backend Running"}