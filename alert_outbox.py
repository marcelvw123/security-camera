"""Persistent WhatsApp intent with conservative handling of ambiguous sends."""
import json
import os
import sqlite3
import time
from contextlib import closing
from pathlib import Path


class AlertOutbox:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        descriptor = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600) if not self.path.exists() else None
        if descriptor is not None:
            os.close(descriptor)
        with closing(sqlite3.connect(self.path, timeout=10)) as db, db:
            db.execute('''CREATE TABLE IF NOT EXISTS alerts (
                path TEXT PRIMARY KEY, camera TEXT NOT NULL, labels TEXT NOT NULL,
                status TEXT NOT NULL, updated REAL NOT NULL)''')
            # The provider may have accepted an interrupted request. Never resend it blindly.
            db.execute("UPDATE alerts SET status='unknown', updated=? WHERE status='sending'", (time.time(),))
        self.path.chmod(0o600)

    def enqueue(self, path, camera, labels):
        with closing(sqlite3.connect(self.path, timeout=10)) as db, db:
            db.execute('INSERT OR IGNORE INTO alerts VALUES (?,?,?,?,?)',
                       (str(Path(path).resolve()), camera, json.dumps(labels), 'pending', time.time()))

    def claim(self):
        with closing(sqlite3.connect(self.path, timeout=10)) as db:
            db.execute('BEGIN IMMEDIATE')
            row = db.execute("SELECT path,camera,labels FROM alerts WHERE status='pending' ORDER BY updated LIMIT 1").fetchone()
            if row:
                db.execute("UPDATE alerts SET status='sending', updated=? WHERE path=?", (time.time(), row[0]))
            db.commit()
            return (row[0], row[1], json.loads(row[2])) if row else None

    def finish(self, path, sent):
        with closing(sqlite3.connect(self.path, timeout=10)) as db, db:
            db.execute('UPDATE alerts SET status=?,updated=? WHERE path=? AND status=?',
                       ('accepted' if sent else 'unknown', time.time(), str(Path(path).resolve()), 'sending'))

    def status(self, path):
        with closing(sqlite3.connect(self.path, timeout=10)) as db:
            row = db.execute('SELECT status FROM alerts WHERE path=?', (str(Path(path).resolve()),)).fetchone()
            return row[0] if row else None

    def counts(self):
        with closing(sqlite3.connect(self.path, timeout=10)) as db:
            return dict(db.execute('SELECT status,count(*) FROM alerts GROUP BY status').fetchall())
