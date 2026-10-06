import sqlite3
from datetime import datetime, timezone

class SQLiteDatabase:
    def __init__(self, db_path):
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()
        self.create_table()

    def create_table(self):
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                tenant_id TEXT,
                event_type TEXT,
                val REAL,
                timestamp REAL
            )
        ''')
        self.conn.commit()

    def init_db(self, db_path):
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()
        self.create_table()

    def insert_event(self, tenant_id, event_type, val):
        current_timestamp = datetime.now(timezone.utc).timestamp()
        self.cursor.execute('''
            INSERT INTO events (tenant_id, event_type, val, timestamp)
            VALUES (?, ?, ?, ?)
        ''', (tenant_id, event_type, val, current_timestamp))
        self.conn.commit()

    def get_events(self, tenant_id=None):
        self.cursor.execute('''
            SELECT * FROM events
            WHERE tenant_id = ?
            ORDER BY timestamp
        ''', (tenant_id,))
        rows = self.cursor.fetchall()
        self.conn.close()
        return rows