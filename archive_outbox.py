"""Durable archive queue. A transport must verify remote bytes before returning True.

This module is deliberately not wired into capture until the archive API is ready;
using the older WhatsApp upload endpoint here would risk deleting the only copy.
"""
from contextlib import contextmanager
import hashlib
import json
import os
import sqlite3
import time
import uuid
from pathlib import Path


class ArchiveOutbox:
    def __init__(self, database):
        self.database = Path(database)
        self.database.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        descriptor = os.open(self.database, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600) if not self.database.exists() else None
        if descriptor is not None:
            os.close(descriptor)
        with self.connect() as db:
            db.execute('''CREATE TABLE IF NOT EXISTS archive_jobs (
                id TEXT PRIMARY KEY, path TEXT NOT NULL UNIQUE, kind TEXT NOT NULL,
                metadata TEXT NOT NULL, state TEXT NOT NULL DEFAULT 'pending',
                attempts INTEGER NOT NULL DEFAULT 0, retry_at REAL NOT NULL DEFAULT 0,
                last_error TEXT, retain_local INTEGER NOT NULL DEFAULT 0)''')
        os.chmod(self.database, 0o600)

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.database, timeout=10)
        try:
            db.execute('PRAGMA synchronous=FULL')
            with db:
                yield db
        finally:
            db.close()

    def enqueue(self, path, kind, metadata, retain_local=False):
        if kind not in ('motion', 'object', 'image'):
            raise ValueError('Invalid archive kind')
        path = str(Path(path).resolve(strict=True))
        encoded = json.dumps(metadata, sort_keys=True, separators=(',', ':'))
        with self.connect() as db:
            existing = db.execute('SELECT id, kind, metadata FROM archive_jobs WHERE path=?', (path,)).fetchone()
            if existing:
                if existing[1:] != (kind, encoded):
                    raise ValueError('Archive path already has different metadata')
                return existing[0]
            job_id = str(uuid.uuid4())
            db.execute('INSERT INTO archive_jobs(id,path,kind,metadata,retain_local) VALUES (?,?,?,?,?)',
                       (job_id, path, kind, encoded, int(retain_local)))
            return job_id

    def retry_cleanup(self, path):
        # If the Pi lost power after verification but before unlink, re-verify.
        with self.connect() as db:
            db.execute("UPDATE archive_jobs SET state='pending', retry_at=0 WHERE path=? AND state='verified' AND retain_local=0",
                       (str(Path(path).resolve()),))

    def release(self, path):
        # Re-verify before unlinking, including after a restart.
        with self.connect() as db:
            db.execute("UPDATE archive_jobs SET retain_local=0, state='pending', retry_at=0 WHERE path=?",
                       (str(Path(path).resolve()),))

    def counts(self):
        with self.connect() as db:
            counts = dict(db.execute('SELECT state, count(*) FROM archive_jobs GROUP BY state').fetchall())
            counts['failed'] = db.execute("SELECT count(*) FROM archive_jobs WHERE state='pending' AND last_error IS NOT NULL").fetchone()[0]
            counts['retained'] = db.execute("SELECT count(*) FROM archive_jobs WHERE state='verified' AND retain_local=1").fetchone()[0]
            return counts

    def process_one(self, transport, now=None):
        """Retry jobs after failures; never unlink unless transport confirms verification."""
        now = time.time() if now is None else now
        with self.connect() as db:
            row = db.execute('SELECT id,path,kind,metadata,attempts,retain_local FROM archive_jobs '
                             "WHERE state='pending' AND retry_at<=? ORDER BY rowid LIMIT 1", (now,)).fetchone()
        if not row:
            return None
        job_id, filename, kind, metadata, attempts, retain_local = row
        path = Path(filename)
        try:
            saved_metadata = json.loads(metadata)
            sidecar = Path(str(path) + '.detections.json')
            expected_sidecar = saved_metadata.get('detections')
            if expected_sidecar is not None:
                if json.loads(sidecar.read_text()) != expected_sidecar:
                    raise ValueError('Detection metadata changed')
            before = path.stat()
            digest = hashlib.md5(usedforsecurity=False)
            with path.open('rb') as source:
                while chunk := source.read(1024 * 1024):
                    digest.update(chunk)
            identity = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
            after_hash = path.stat()
            if identity != (after_hash.st_dev, after_hash.st_ino, after_hash.st_size, after_hash.st_mtime_ns):
                raise ValueError('File changed while hashing')
            if transport(job_id, path, kind, json.loads(metadata), digest.digest(), before.st_size) is not True:
                raise ValueError('Remote verification did not succeed')
            current = path.stat()
            if identity != (current.st_dev, current.st_ino, current.st_size, current.st_mtime_ns):
                raise ValueError('File changed before deletion')
            if expected_sidecar is not None and json.loads(sidecar.read_text()) != expected_sidecar:
                raise ValueError('Detection metadata changed before deletion')
            # Commit verified state first. A crash before unlink retains an extra copy.
            with self.connect() as db:
                db.execute("UPDATE archive_jobs SET state='verified', last_error=NULL WHERE id=?", (job_id,))
            if not retain_local:
                path.unlink()
                if expected_sidecar is not None:
                    sidecar.unlink()
            return 'verified'
        except Exception as exc:
            # Never store exception text: providers may include SAS URLs in errors.
            with self.connect() as db:
                db.execute('UPDATE archive_jobs SET attempts=attempts+1, retry_at=?, last_error=? WHERE id=?',
                           (now + min(3600, 10 * 2 ** min(attempts, 8)), type(exc).__name__, job_id))
            return 'retry'
