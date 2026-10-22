
import sqlite3
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DB_PATH = DATA_DIR / "lifeline.db"


def get_connection():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def initialize_database():
    with get_connection() as conn:
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS blood_banks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            facility_id TEXT NOT NULL UNIQUE,
            facility_name TEXT NOT NULL,
            category TEXT,
            address TEXT,
            phone TEXT,
            email TEXT
        );

        CREATE TABLE IF NOT EXISTS inventory (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            row_hash TEXT NOT NULL UNIQUE,
            facility_id TEXT,
            facility_name TEXT,
            component TEXT,
            blood_group TEXT,
            units_listed TEXT,
            reported_status TEXT,
            last_updated TEXT,
            source_file TEXT NOT NULL,
            imported_at TEXT DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS component_status (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            row_hash TEXT NOT NULL UNIQUE,
            facility_id TEXT,
            facility_name TEXT,
            component TEXT,
            reported_status TEXT,
            last_updated TEXT,
            source_file TEXT NOT NULL,
            imported_at TEXT DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS import_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_file TEXT NOT NULL,
            rows_seen INTEGER NOT NULL,
            rows_inserted INTEGER NOT NULL,
            imported_at TEXT DEFAULT CURRENT_TIMESTAMP
        );
        """)


def get_inventory_records():
    """Return saved inventory joined with known facility details."""
    query = """
        SELECT
            i.facility_id AS "Facility ID",
            COALESCE(
                NULLIF(i.facility_name, ''),
                b.facility_name,
                ''
            ) AS "Blood Bank",
            i.blood_group AS "Blood Group",
            i.component AS "Component",
            i.units_listed AS "Listed Units",
            i.reported_status AS "Source Status",
            i.last_updated AS "Snapshot Date",
            COALESCE(b.address, '') AS "Address",
            COALESCE(b.phone, '') AS "Phone",
            COALESCE(b.email, '') AS "Email",
            COALESCE(b.category, '') AS "Category",
            COALESCE(b.address, '') AS "Location",
            i.source_file AS "Source File",
            i.imported_at AS "Imported At"
        FROM inventory AS i
        LEFT JOIN blood_banks AS b
            ON i.facility_id = b.facility_id
        ORDER BY "Blood Bank", "Blood Group", "Component"
    """

    with get_connection() as conn:
        return pd.read_sql_query(query, conn)


def get_blood_banks():
    query = """
        SELECT facility_id, facility_name, category,
               address, phone, email
        FROM blood_banks
        ORDER BY facility_name
    """

    with get_connection() as conn:
        return pd.read_sql_query(query, conn)


def get_counts():
    with get_connection() as conn:
        return {
            "blood_banks": conn.execute(
                "SELECT COUNT(*) FROM blood_banks"
            ).fetchone()[0],
            "inventory": conn.execute(
                "SELECT COUNT(*) FROM inventory"
            ).fetchone()[0],
            "component_status": conn.execute(
                "SELECT COUNT(*) FROM component_status"
            ).fetchone()[0],
        }
