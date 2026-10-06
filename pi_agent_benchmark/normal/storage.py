import sqlite3
from pathlib import Path

class EventStorage:
    def __init__(self, db_path):
        self.db_path = db_path
        self.conn = None
        self.create_table()

    def create_table(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    tenant_id TEXT,
                    event_type TEXT,
                    val REAL,
                    timestamp REAL
                )'''
            )

    def init_db(self, db_path):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.create_table()

    def insert_event(self, tenant_id, event_type, val):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO events (tenant_id, event_type, val, timestamp)
                VALUES (?, ?, ?, ?)
            ''', (tenant_id, event_type, val, datetime.now().timestamp()))
            conn.commit()

    def get_events(self, tenant_id=None):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT * FROM events
                WHERE tenant_id = ?
                ORDER BY timestamp
            ''', (tenant_id,)) if tenant_id else '''
                SELECT * FROM events
                ORDER BY timestamp
            '''
            events = cursor.fetchall()
            conn.close()
            return events