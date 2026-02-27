from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import psycopg2
import os
import io
import pandas as pd
from prophet import Prophet

app = Flask(__name__)
CORS(app)


# -----------------------------
# DATABASE CONNECTION
# -----------------------------
def get_connection():
    url = os.environ.get("DATABASE_URL")

    if not url:
        raise Exception("DATABASE_URL not set")

    if url.startswith("postgres://"):
        url = url.replace("postgres://", "postgresql://", 1)

    return psycopg2.connect(url)


# -----------------------------
# INIT DB (Manual)
# -----------------------------
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


# -----------------------------
# AREAS
# -----------------------------
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


# -----------------------------
# ADD READING
# -----------------------------
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

    prev_domestic = prev[0] if prev else 0
    prev_flush = prev[1] if prev else 0

    domestic_usage = domestic_today - prev_domestic
    flush_usage = flush_today - prev_flush
    total_usage = domestic_usage + flush_usage

    anomaly = 1 if total_usage > 500 else 0

    cur.execute("""
        INSERT INTO readings (
            hostel_name, date,
            domestic_reading, flush_reading,
            domestic_usage, flush_usage,
            total_usage, anomaly_flag
        )
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
    """, (
        hostel, date_val,
        domestic_today, flush_today,
        domestic_usage, flush_usage,
        total_usage, anomaly
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


# -----------------------------
# DASHBOARD (All Areas Always)
# -----------------------------
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
    cur.close()
    conn.close()

    data_map = {row[0]: row for row in rows}

    areas_data = []
    total_today = 0
    total_domestic = 0
    total_flush = 0

    for area in AREAS:
        if area in data_map:
            row = data_map[area]
            domestic = row[1]
            flush = row[2]
            total = row[3]
            anomaly = row[4]
        else:
            domestic = 0
            flush = 0
            total = 0
            anomaly = 0

        areas_data.append({
            "hostel_name": area,
            "domestic_usage": domestic,
            "flush_usage": flush,
            "total_usage": total,
            "anomaly_flag": anomaly
        })

        total_domestic += domestic
        total_flush += flush
        total_today += total

    top_areas = sorted(
        areas_data,
        key=lambda x: x["total_usage"],
        reverse=True
    )[:3]

    return jsonify({
        "areas": areas_data,
        "total_today": total_today,
        "total_domestic": total_domestic,
        "total_flush": total_flush,
        "top_areas": top_areas
    })


from datetime import datetime, timedelta
import numpy as np

@app.route("/trend")
def get_trend():

    areas_param = request.args.get("areas")

    if not areas_param:
        return jsonify({"data": []})

    areas = areas_param.split(",")

    conn = get_connection()
    cur = conn.cursor()

    # Get historical data for selected areas
    cur.execute("""
        SELECT date, hostel_name, domestic_usage, flush_usage
        FROM readings
        WHERE hostel_name = ANY(%s)
        ORDER BY date ASC
    """, (areas,))

    rows = cur.fetchall()

    if not rows:
        return jsonify({"data": []})

    # Group by date
    historical = {}

    for row in rows:
        date = row[0]
        domestic = row[2]
        flush = row[3]

        if date not in historical:
            historical[date] = {"domestic": 0, "flush": 0}

        historical[date]["domestic"] += domestic
        historical[date]["flush"] += flush

    sorted_dates = sorted(historical.keys())

    trend_data = []

    # Add historical data
    for date in sorted_dates:
        trend_data.append({
            "date": date.strftime("%Y-%m-%d"),
            "domestic": historical[date]["domestic"],
            "flush": historical[date]["flush"],
            "is_forecast": False
        })

    # -------- FORECAST SECTION --------

    latest_date = sorted_dates[-1]

    # Use last 7 days for simple linear trend
    last_7 = sorted_dates[-7:] if len(sorted_dates) >= 7 else sorted_dates

    x = np.arange(len(last_7))

    domestic_vals = [historical[d]["domestic"] for d in last_7]
    flush_vals = [historical[d]["flush"] for d in last_7]

    # Linear regression
    domestic_coef = np.polyfit(x, domestic_vals, 1)
    flush_coef = np.polyfit(x, flush_vals, 1)

    # Predict next 3 days
    for i in range(1, 4):

        next_date = latest_date + timedelta(days=i)
        future_x = len(last_7) + i - 1

        domestic_pred = domestic_coef[0] * future_x + domestic_coef[1]
        flush_pred = flush_coef[0] * future_x + flush_coef[1]

        trend_data.append({
            "date": next_date.strftime("%Y-%m-%d"),
            "domestic": round(max(domestic_pred, 0), 2),
            "flush": round(max(flush_pred, 0), 2),
            "is_forecast": True
        })

    cur.close()
    conn.close()

    return jsonify({"data": trend_data})

# -----------------------------
# EXPORT (Structured)
# -----------------------------
@app.route("/export", methods=["GET"])
def export_data():

    start_date = request.args.get("start_date")
    end_date = request.args.get("end_date")

    conn = get_connection()

    df = pd.read_sql("""
        SELECT date,
               hostel_name,
               domestic_usage,
               flush_usage
        FROM readings
        WHERE date BETWEEN %s AND %s
        ORDER BY date ASC
    """, conn, params=(start_date, end_date))

    conn.close()

    if df.empty:
        return {"error": "No data found"}, 404

    pivot_domestic = df.pivot_table(
        index="date",
        columns="hostel_name",
        values="domestic_usage",
        aggfunc="sum"
    ).fillna(0)

    pivot_flush = df.pivot_table(
        index="date",
        columns="hostel_name",
        values="flush_usage",
        aggfunc="sum"
    ).fillna(0)

    for area in AREAS:
        if area not in pivot_domestic.columns:
            pivot_domestic[area] = 0
        if area not in pivot_flush.columns:
            pivot_flush[area] = 0

    pivot_domestic = pivot_domestic[AREAS]
    pivot_flush = pivot_flush[AREAS]

    combined = pd.DataFrame()
    combined["Date"] = pivot_domestic.index

    for area in AREAS:
        combined[f"{area} F"] = pivot_flush[area].values
        combined[f"{area} D"] = pivot_domestic[area].values

    output = io.BytesIO()

    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        combined.to_excel(writer, index=False)

    output.seek(0)

    return send_file(
        output,
        as_attachment=True,
        download_name="structured_water_report.xlsx",
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )


@app.route("/")
def home():
    return {"message": "Full Water Intelligence Backend Running"}