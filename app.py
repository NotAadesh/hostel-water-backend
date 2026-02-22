from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import psycopg2
import os
import io
import pandas as pd

app = Flask(__name__)
CORS(app)


# ----------------------------------
# DATABASE CONNECTION (RENDER SAFE)
# ----------------------------------
def get_connection():
    url = os.environ.get("DATABASE_URL")

    if not url:
        raise Exception("DATABASE_URL not set")

    # Fix Render postgres URL format
    if url.startswith("postgres://"):
        url = url.replace("postgres://", "postgresql://", 1)

    return psycopg2.connect(url)


# ----------------------------------
# INIT DATABASE (MANUAL RESET ONLY)
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


# ----------------------------------
# MANUAL RESET ROUTE
# ----------------------------------
@app.route("/reset_db")
def reset_db():
    init_db()
    return {"message": "Database reset complete"}


# ----------------------------------
# AREAS LIST
# ----------------------------------
AREAS = [
    "HOSTEL 1", "HOSTEL 2", "HOSTEL 3", "HOSTEL 4",
    "HOSTEL 5", "HOSTEL 6", "HOSTEL 7", "HOSTEL 8",
    "HOSTEL 9", "HOSTEL 10",
    "CAFETERIA 1", "CAFETERIA 2",
    "ACADEMIC BLOCK",
    "NOB",
    "HOUSING FACILITY 1",
    "HOUSING FACILITY 2",
    "HOUSING FACILITY 3",
    "HOUSING FACILITY 4",
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

    # Get previous readings
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
        SELECT hostel_name, total_usage, anomaly_flag
        FROM readings
        WHERE date = (SELECT MAX(date) FROM readings)
    """)

    rows = cur.fetchall()

    areas_data = []
    total_today = 0

    for row in rows:
        areas_data.append({
            "hostel_name": row[0],
            "total_usage": row[1],
            "anomaly_flag": row[2]
        })
        total_today += row[1]

    cur.close()
    conn.close()

    return jsonify({
        "areas": areas_data,
        "total_today": total_today
    })


# ----------------------------------
# STRUCTURED EXPORT (STABLE)
# ----------------------------------
@app.route("/export", methods=["GET"])
def export_data():
    start_date = request.args.get("start_date")
    end_date = request.args.get("end_date")

    if not start_date or not end_date:
        return {"error": "Start and end date required"}, 400

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
    )

    pivot_flush = df.pivot_table(
        index="date",
        columns="hostel_name",
        values="flush_usage",
        aggfunc="sum"
    )

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
        combined[f"{area} F"] = pivot_flush[area].fillna(0).values
        combined[f"{area} D"] = pivot_domestic[area].fillna(0).values

    output = io.BytesIO()

    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        combined.to_excel(writer, sheet_name="Water Report", index=False)

    output.seek(0)

    return send_file(
        output,
        as_attachment=True,
        download_name="structured_water_report.xlsx",
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )


# ----------------------------------
# ROOT
# ----------------------------------
@app.route("/")
def home():
    return {"message": "Domestic + Flush Water Backend Running"}