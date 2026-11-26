"""
storage.py - Dual-backend persistence (SQLite default, MongoDB Atlas ready).

Both annotator_server.py and update_clusters.py use this module so the app runs
out of the box with zero external services locally (SQLite), while keeping full
MongoDB Atlas compatibility intact for multi-user cloud deployments.

Backend selection order:
  1. Explicit backend='mongo' or MongoDB credentials supplied
     (MONGO_URI or MONGO_USERNAME + MONGO_PASSWORD + MONGO_HOST) -> MongoStore
  2. Otherwise -> SQLiteStore (stored at data/annotator.db by default)
"""
import os
import json
import sqlite3
import threading
from datetime import datetime
from urllib.parse import quote_plus


def resolve_mongo_uri(mongo_uri: str = '',
                      mongo_username: str = '',
                      mongo_password: str = '',
                      mongo_host: str = '') -> str:
    if mongo_uri:
        return mongo_uri
    if mongo_username and mongo_password and mongo_host:
        return (
            f'mongodb+srv://{quote_plus(mongo_username)}:{quote_plus(mongo_password)}'
            f'@{mongo_host}/?retryWrites=true&w=majority&appName=Cluster0'
        )
    return ''


def _json_safe(obj):
    if isinstance(obj, (bool, int, str)) or obj is None:
        return obj
    if isinstance(obj, float):
        return obj
    try:
        import numpy as np
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return _json_safe(obj.tolist())
    except ImportError:
        pass
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    return str(obj)


# ---------------------------------------------------------------------------
# SQLite backend (zero-config local / single-server storage)
# ---------------------------------------------------------------------------

class SQLiteStore:
    backend = 'sqlite'

    def __init__(self, db_path: str):
        self.db_path = os.path.abspath(db_path)
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._lock = threading.RLock()
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30.0, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute('PRAGMA journal_mode=WAL;')
        conn.execute('PRAGMA synchronous=NORMAL;')
        return conn

    def _init_schema(self) -> None:
        with self._lock, self._connect() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS users (
                    username   TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS versions (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    user            TEXT NOT NULL,
                    slide           TEXT NOT NULL,
                    timestamp       TEXT NOT NULL,
                    tile_nums       TEXT NOT NULL,
                    marked_count    INTEGER NOT NULL,
                    embedding_xy    TEXT NOT NULL,
                    associated_data TEXT NOT NULL,
                    actions_log     TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_versions_user_slide_ts
                    ON versions (user, slide, timestamp DESC);

                CREATE TABLE IF NOT EXISTS umap_embeddings (
                    username   TEXT NOT NULL,
                    slide      TEXT NOT NULL,
                    tile_nums  TEXT NOT NULL,
                    embeddings TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (username, slide)
                );

                CREATE TABLE IF NOT EXISTS features_assigned (
                    user          TEXT NOT NULL,
                    slide         TEXT NOT NULL,
                    tile_nums     TEXT NOT NULL,
                    samples       TEXT NOT NULL,
                    labels        TEXT NOT NULL,
                    probabilities TEXT NOT NULL,
                    updated_at    TEXT NOT NULL,
                    PRIMARY KEY (user, slide)
                );

                CREATE TABLE IF NOT EXISTS user_annotations (
                    user       TEXT NOT NULL,
                    slide      TEXT NOT NULL,
                    tile_num   INTEGER NOT NULL,
                    labels     TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (user, slide, tile_num)
                );
                CREATE INDEX IF NOT EXISTS idx_user_ann_user_slide
                    ON user_annotations (user, slide);
            """)

    def ping(self) -> None:
        with self._lock, self._connect() as conn:
            conn.execute('SELECT 1').fetchone()

    def close(self) -> None:
        pass

    def ensure_user(self, username: str) -> bool:
        now = datetime.now().isoformat(timespec='seconds')
        with self._lock, self._connect() as conn:
            cur = conn.execute(
                'INSERT OR IGNORE INTO users (username, created_at) VALUES (?, ?)',
                (username, now),
            )
            return cur.rowcount > 0

    def list_versions(self, username: str, slide_name: str, limit: int = 200) -> list[dict]:
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                """SELECT id, timestamp, marked_count
                   FROM versions
                   WHERE user = ? AND slide = ?
                   ORDER BY timestamp DESC, id DESC
                   LIMIT ?""",
                (username, slide_name, limit),
            ).fetchall()
        return [{
            'id': str(r['id']),
            'timestamp': r['timestamp'],
            'marked_count': int(r['marked_count']),
        } for r in rows]

    def save_version(self, username: str, slide_name: str, tile_nums: list[int],
                     marked_count: int, embedding_xy: dict, associated_data: dict,
                     actions_log: list) -> dict:
        ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        with self._lock, self._connect() as conn:
            cur = conn.execute(
                """INSERT INTO versions
                   (user, slide, timestamp, tile_nums, marked_count,
                    embedding_xy, associated_data, actions_log)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    username, slide_name, ts,
                    json.dumps(_json_safe(tile_nums)),
                    int(marked_count),
                    json.dumps(_json_safe(embedding_xy)),
                    json.dumps(_json_safe(associated_data)),
                    json.dumps(_json_safe(actions_log)),
                ),
            )
            vid = str(cur.lastrowid)
        return {'id': vid, 'timestamp': ts, 'marked_count': int(marked_count)}

    def get_version(self, version_id: str, username: str, slide_name: str) -> dict | None:
        try:
            vid = int(version_id)
        except (TypeError, ValueError):
            return None
        with self._lock, self._connect() as conn:
            r = conn.execute(
                """SELECT id, timestamp, tile_nums, embedding_xy, associated_data
                   FROM versions
                   WHERE id = ? AND user = ? AND slide = ?""",
                (vid, username, slide_name),
            ).fetchone()
        if not r:
            return None
        return {
            'id': str(r['id']),
            'timestamp': r['timestamp'],
            'tile_nums': json.loads(r['tile_nums']),
            'embeddingXY': json.loads(r['embedding_xy']),
            'associated_data': json.loads(r['associated_data']),
        }

    def get_latest_version(self, username: str, slide_name: str) -> dict | None:
        with self._lock, self._connect() as conn:
            r = conn.execute(
                """SELECT id, timestamp, tile_nums, embedding_xy, associated_data
                   FROM versions
                   WHERE user = ? AND slide = ?
                   ORDER BY timestamp DESC, id DESC
                   LIMIT 1""",
                (username, slide_name),
            ).fetchone()
        if not r:
            return None
        return {
            'id': str(r['id']),
            'timestamp': r['timestamp'],
            'tile_nums': json.loads(r['tile_nums']),
            'embeddingXY': json.loads(r['embedding_xy']),
            'associated_data': json.loads(r['associated_data']),
        }

    def get_umap_embeddings(self, username: str, slide_name: str) -> dict | None:
        with self._lock, self._connect() as conn:
            r = conn.execute(
                'SELECT tile_nums, embeddings FROM umap_embeddings WHERE username = ? AND slide = ?',
                (username, slide_name),
            ).fetchone()
        if not r:
            return None
        return {
            'tile_nums': json.loads(r['tile_nums']),
            'embeddings': json.loads(r['embeddings']),
        }

    def save_umap_embeddings(self, username: str, slide_name: str,
                             tile_nums: list[int], embeddings: list) -> None:
        now = datetime.now().isoformat(timespec='seconds')
        with self._lock, self._connect() as conn:
            conn.execute(
                """INSERT INTO umap_embeddings (username, slide, tile_nums, embeddings, updated_at)
                   VALUES (?, ?, ?, ?, ?)
                   ON CONFLICT(username, slide) DO UPDATE SET
                       tile_nums = excluded.tile_nums,
                       embeddings = excluded.embeddings,
                       updated_at = excluded.updated_at""",
                (username, slide_name,
                 json.dumps(_json_safe(tile_nums)),
                 json.dumps(_json_safe(embeddings)),
                 now),
            )

    def get_features_assigned(self, username: str, slide_name: str) -> dict | None:
        with self._lock, self._connect() as conn:
            r = conn.execute(
                """SELECT tile_nums, samples, labels, probabilities
                   FROM features_assigned WHERE user = ? AND slide = ?""",
                (username, slide_name),
            ).fetchone()
        if not r:
            return None
        return {
            'tile_nums': json.loads(r['tile_nums']),
            'samples': json.loads(r['samples']),
            'labels': json.loads(r['labels']),
            'probabilities': json.loads(r['probabilities']),
        }

    def save_features_assigned(self, username: str, slide_name: str,
                               tile_nums: list[int], samples: list[int],
                               labels: list[str], probabilities: list[str]) -> None:
        now = datetime.now().isoformat(timespec='seconds')
        with self._lock, self._connect() as conn:
            conn.execute(
                """INSERT INTO features_assigned
                   (user, slide, tile_nums, samples, labels, probabilities, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(user, slide) DO UPDATE SET
                       tile_nums = excluded.tile_nums,
                       samples = excluded.samples,
                       labels = excluded.labels,
                       probabilities = excluded.probabilities,
                       updated_at = excluded.updated_at""",
                (username, slide_name,
                 json.dumps(_json_safe(tile_nums)),
                 json.dumps(_json_safe(samples)),
                 json.dumps(_json_safe(labels)),
                 json.dumps(_json_safe(probabilities)),
                 now),
            )

    def save_user_annotation(self, username: str, slide_name: str,
                             tile_num: int, labels: dict) -> None:
        now = datetime.now().isoformat(timespec='seconds')
        with self._lock, self._connect() as conn:
            conn.execute(
                """INSERT INTO user_annotations (user, slide, tile_num, labels, updated_at)
                   VALUES (?, ?, ?, ?, ?)
                   ON CONFLICT(user, slide, tile_num) DO UPDATE SET
                       labels = excluded.labels,
                       updated_at = excluded.updated_at""",
                (username, slide_name, int(tile_num), json.dumps(_json_safe(labels)), now),
            )

    def delete_user_annotation(self, slide_name: str, tile_num: int) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                'DELETE FROM user_annotations WHERE slide = ? AND tile_num = ?',
                (slide_name, int(tile_num)),
            )

    def get_user_annotations(self, username: str, slide_name: str) -> dict[int, dict]:
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                'SELECT tile_num, labels FROM user_annotations WHERE user = ? AND slide = ?',
                (username, slide_name),
            ).fetchall()
        return {int(r['tile_num']): json.loads(r['labels']) for r in rows}


# ---------------------------------------------------------------------------
# MongoDB backend (preserves existing Atlas schema for cloud hosting)
# ---------------------------------------------------------------------------

class MongoStore:
    backend = 'mongodb'

    def __init__(self, uri: str, db_name: str):
        from pymongo import ASCENDING, DESCENDING, MongoClient
        self._ASC = ASCENDING
        self._DESC = DESCENDING
        self._client = MongoClient(uri, serverSelectionTimeoutMS=8000)
        self.db = self._client[db_name]
        self._ensure_indexes()

    def _ensure_indexes(self) -> None:
        specs = [
            (self.db.versions, [('user', self._ASC), ('slide', self._ASC), ('timestamp', self._DESC)], {}),
            (self.db.umap_embeddings, [('username', self._ASC), ('slide', self._ASC)], {}),
            (self.db.features_assigned, [('user', self._ASC), ('slide', self._ASC)], {}),
            (self.db.user_annotations, [('user', self._ASC), ('slide', self._ASC), ('tile_num', self._ASC)], {'unique': True}),
            (self.db.users, [('username', self._ASC)], {'unique': True}),
        ]
        for coll, keys, opts in specs:
            try:
                coll.create_index(keys, **opts)
            except Exception:
                pass

    def ping(self) -> None:
        self.db.command('ping')

    def close(self) -> None:
        self._client.close()

    def ensure_user(self, username: str) -> bool:
        res = self.db.users.update_one(
            {'username': username}, {'$setOnInsert': {'username': username}}, upsert=True
        )
        return res.upserted_id is not None

    def list_versions(self, username: str, slide_name: str, limit: int = 200) -> list[dict]:
        pipeline = [
            {'$match': {'user': username, 'slide': slide_name}},
            {'$sort': {'timestamp': -1}},
            {'$limit': limit},
            {'$project': {
                'timestamp': 1,
                'marked_count': {'$ifNull': [
                    '$marked_count',
                    {'$sum': {'$ifNull': ['$embeddingXY.marked', []]}},
                ]},
            }},
        ]
        out = []
        for v in self.db.versions.aggregate(pipeline):
            ts = v['timestamp']
            ts_str = ts.strftime('%Y-%m-%d %H:%M:%S') if hasattr(ts, 'strftime') else str(ts)
            out.append({
                'id': str(v['_id']),
                'timestamp': ts_str,
                'marked_count': int(v.get('marked_count') or 0),
            })
        return out

    def save_version(self, username: str, slide_name: str, tile_nums: list[int],
                     marked_count: int, embedding_xy: dict, associated_data: dict,
                     actions_log: list) -> dict:
        now = datetime.now()
        doc = {
            'user': username,
            'slide': slide_name,
            'timestamp': now,
            'tile_nums': _json_safe(tile_nums),
            'marked_count': int(marked_count),
            'embeddingXY': _json_safe(embedding_xy),
            'associated_data': _json_safe(associated_data),
            'actions_log': _json_safe(actions_log),
        }
        res = self.db.versions.insert_one(doc)
        return {
            'id': str(res.inserted_id),
            'timestamp': now.strftime('%Y-%m-%d %H:%M:%S'),
            'marked_count': int(marked_count),
        }

    def get_version(self, version_id: str, username: str, slide_name: str) -> dict | None:
        from bson import ObjectId
        from bson.errors import InvalidId
        try:
            vid = ObjectId(str(version_id))
        except (InvalidId, TypeError):
            return None
        v = self.db.versions.find_one(
            {'_id': vid, 'user': username, 'slide': slide_name},
            projection={'embeddingXY.marked': 1, 'associated_data': 1, 'tile_nums': 1, 'timestamp': 1},
        )
        if not v:
            return None
        ts = v['timestamp']
        return {
            'id': str(v['_id']),
            'timestamp': ts.strftime('%Y-%m-%d %H:%M:%S') if hasattr(ts, 'strftime') else str(ts),
            'tile_nums': v.get('tile_nums'),
            'embeddingXY': v.get('embeddingXY') or {},
            'associated_data': v.get('associated_data') or {},
        }

    def get_latest_version(self, username: str, slide_name: str) -> dict | None:
        v = self.db.versions.find_one(
            {'user': username, 'slide': slide_name},
            sort=[('timestamp', self._DESC)],
            projection={'embeddingXY.marked': 1, 'associated_data': 1, 'tile_nums': 1, 'timestamp': 1},
        )
        if not v:
            return None
        ts = v.get('timestamp')
        return {
            'id': str(v['_id']),
            'timestamp': ts.strftime('%Y-%m-%d %H:%M:%S') if hasattr(ts, 'strftime') else str(ts),
            'tile_nums': v.get('tile_nums'),
            'embeddingXY': v.get('embeddingXY') or {},
            'associated_data': v.get('associated_data') or {},
        }

    def get_umap_embeddings(self, username: str, slide_name: str) -> dict | None:
        doc = self.db.umap_embeddings.find_one({'username': username, 'slide': slide_name})
        if not doc or not doc.get('embeddings'):
            return None
        return {'tile_nums': doc.get('tile_nums'), 'embeddings': doc['embeddings']}

    def save_umap_embeddings(self, username: str, slide_name: str,
                             tile_nums: list[int], embeddings: list) -> None:
        self.db.umap_embeddings.update_one(
            {'username': username, 'slide': slide_name},
            {'$set': {
                'tile_nums': _json_safe(tile_nums),
                'embeddings': _json_safe(embeddings),
                'updated_at': datetime.now(),
            }},
            upsert=True,
        )

    def get_features_assigned(self, username: str, slide_name: str) -> dict | None:
        doc = self.db.features_assigned.find_one({'user': username, 'slide': slide_name})
        if not doc:
            return None
        return {
            'tile_nums': doc.get('tile_nums'),
            'samples': doc.get('samples', []),
            'labels': doc.get('labels', []),
            'probabilities': doc.get('probabilities', []),
        }

    def save_features_assigned(self, username: str, slide_name: str,
                               tile_nums: list[int], samples: list[int],
                               labels: list[str], probabilities: list[str]) -> None:
        self.db.features_assigned.update_one(
            {'user': username, 'slide': slide_name},
            {'$set': {
                'user': username,
                'slide': slide_name,
                'tile_nums': _json_safe(tile_nums),
                'samples': _json_safe(samples),
                'labels': _json_safe(labels),
                'probabilities': _json_safe(probabilities),
                'updated_at': datetime.now(),
            }},
            upsert=True,
        )

    def save_user_annotation(self, username: str, slide_name: str,
                             tile_num: int, labels: dict) -> None:
        self.db.user_annotations.update_one(
            {'user': username, 'slide': slide_name, 'tile_num': int(tile_num)},
            {'$set': {'labels': _json_safe(labels), 'updated_at': datetime.now()}},
            upsert=True,
        )

    def delete_user_annotation(self, slide_name: str, tile_num: int) -> None:
        self.db.user_annotations.delete_many({'slide': slide_name, 'tile_num': int(tile_num)})

    def get_user_annotations(self, username: str, slide_name: str) -> dict[int, dict]:
        docs = self.db.user_annotations.find(
            {'user': username, 'slide': slide_name},
            projection={'tile_num': 1, 'labels': 1},
        )
        return {int(d['tile_num']): d['labels'] for d in docs}


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_store(sqlite_path: str = 'data/annotator.db',
                 mongo_uri: str = '',
                 mongo_username: str = '',
                 mongo_password: str = '',
                 mongo_host: str = '',
                 mongo_db: str = 'annotator_db',
                 backend: str = 'auto'):
    uri = resolve_mongo_uri(mongo_uri, mongo_username, mongo_password, mongo_host)
    if backend == 'mongo':
        if not uri:
            raise ValueError('MongoDB backend requested, but no MONGO_URI or credentials provided.')
        return MongoStore(uri, mongo_db)
    if backend == 'sqlite':
        return SQLiteStore(sqlite_path)
    if uri:
        return MongoStore(uri, mongo_db)
    return SQLiteStore(sqlite_path)
