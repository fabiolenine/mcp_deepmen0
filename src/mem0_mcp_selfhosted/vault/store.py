"""Vault persistence: users, tokens and the operation log (stdlib only).

Design invariants:

- **One transaction per mutation, audit included.** Every write goes through
  ``_tx()`` (``BEGIN IMMEDIATE``) and the audit row is inserted inside it, so a
  crash between "token revoked" and "revocation recorded" is impossible: both
  land or neither does.
- **One successor per token.** ``UNIQUE INDEX ... WHERE renewed_from IS NOT
  NULL`` makes concurrent rotation of the same token lose cleanly instead of
  forking the chain.
- **Never the plaintext.** Only ``sha256(token)`` and the 12-char prefix are
  stored; the audit log references the prefix.
- **Schema versioned by ``PRAGMA user_version``**, created idempotently by
  whichever process opens the file first. A file written by a *newer* schema
  raises ``SchemaIncompatible`` instead of being silently misread.

Connections are autocommit (``isolation_level=None``) and opened per
operation — cheap at vault rates (a handful of writes a day, one indexed read
per authorized MCP request) and safe across threads.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterator

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 4

TOKEN_PREFIX = "dm0_"
TOKEN_BODY_LEN = 43  # secrets.token_urlsafe(32)
TOKEN_LEN = len(TOKEN_PREFIX) + TOKEN_BODY_LEN
PREFIX_LEN = 12  # what the UI shows after the one-time reveal
MAX_AUTH_HEADER_LEN = 512  # cap before any parsing work

_TOKEN_RE = re.compile(r"^dm0_[A-Za-z0-9_-]{43}\Z")

# Verification outcomes (grossly reported to the client, precisely logged).
OK = "ok"
MALFORMED = "malformed"
UNKNOWN = "unknown"
REVOKED = "revoked"
EXPIRED = "expired"
USER_DISABLED = "user_disabled"

_SCHEMA_STATEMENTS = (
    """
    CREATE TABLE IF NOT EXISTS users (
      id            INTEGER PRIMARY KEY AUTOINCREMENT,
      email         TEXT    NOT NULL UNIQUE,
      display_name  TEXT    NOT NULL DEFAULT '',
      is_admin      INTEGER NOT NULL DEFAULT 0,
      password_hash TEXT,
      mem0_user_id  TEXT    NOT NULL DEFAULT '',
      created_at    TEXT    NOT NULL,
      disabled_at   TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS tokens (
      id           INTEGER PRIMARY KEY AUTOINCREMENT,
      user_id      INTEGER NOT NULL REFERENCES users(id),
      label        TEXT    NOT NULL DEFAULT '',
      token_hash   TEXT    NOT NULL UNIQUE,
      prefix       TEXT    NOT NULL,
      created_at   TEXT    NOT NULL,
      expires_at   TEXT,
      last_used_at TEXT,
      revoked_at   TEXT,
      renewed_from INTEGER REFERENCES tokens(id)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_tokens_user ON tokens(user_id)",
    # one successor per token: concurrent rotation loses cleanly
    """
    CREATE UNIQUE INDEX IF NOT EXISTS idx_tokens_successor
      ON tokens(renewed_from) WHERE renewed_from IS NOT NULL
    """,
    """
    CREATE TABLE IF NOT EXISTS audit_log (
      id           INTEGER PRIMARY KEY AUTOINCREMENT,
      ts           TEXT    NOT NULL,
      actor_id     INTEGER,
      actor_email  TEXT    NOT NULL DEFAULT '',
      ip           TEXT    NOT NULL DEFAULT '',
      action       TEXT    NOT NULL,
      subject_type TEXT    NOT NULL DEFAULT '',
      subject_id   TEXT    NOT NULL DEFAULT '',
      success      INTEGER NOT NULL DEFAULT 1,
      details      TEXT
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_audit_ts ON audit_log(id DESC)",
    # Durable rollout evidence (schema v2). Throttled logs vanish on restart and
    # cannot answer "was this client silent or was it denied?" — the shadow->on
    # decision needs numbers that survive, bucketed so a flood of 401s cannot
    # turn into a flood of writes.
    """
    CREATE TABLE IF NOT EXISTS auth_denials (
      bucket_start TEXT    NOT NULL,
      reason       TEXT    NOT NULL,
      count        INTEGER NOT NULL DEFAULT 0,
      last_seen_at TEXT    NOT NULL,
      last_path    TEXT    NOT NULL DEFAULT '',
      last_client  TEXT    NOT NULL DEFAULT '',
      PRIMARY KEY (bucket_start, reason)
    )
    """,
)

#: Additive column migrations, applied idempotently: (table, column, DDL type).
_COLUMN_MIGRATIONS = (
    ("tokens", "use_count", "INTEGER NOT NULL DEFAULT 0"),
    # Cliente que roda por demanda (harness de eval, script de manutenção) e
    # NÃO deve ser esperado continuamente. Sem isto, um token legítimo que roda
    # uma vez por semana deixaria promotion_readiness em NOT READY para sempre
    # — a alternativa seria não dar token a ele, que é pior.
    ("tokens", "on_demand", "INTEGER NOT NULL DEFAULT 0"),
    # Época de sessão: a sessão da UI é cookie ASSINADO NO CLIENTE, então
    # logout só apaga a cópia do navegador — uma cópia roubada continuaria
    # valendo as 12h inteiras. O cookie carrega a época; sair (ou trocar a
    # senha) incrementa a do usuário e mata TODAS as sessões dele de uma vez.
    ("users", "session_epoch", "INTEGER NOT NULL DEFAULT 0"),
)

DENIAL_BUCKET_S = 300


class VaultError(Exception):
    """Base class for vault storage failures."""


class SchemaIncompatible(VaultError):
    """The database was written by a newer schema than this code understands."""


class TokenCollision(VaultError):
    """A freshly generated token hash already exists — caller regenerates."""


class SuccessorExists(VaultError):
    """This token was already rotated. Retrying with a new secret cannot help."""


class DuplicateEmail(VaultError):
    """Email already registered."""


class NotFound(VaultError):
    """Referenced user/token does not exist."""


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _iso(dt: datetime) -> str:
    return dt.isoformat()


def hash_token(token: str) -> str:
    """sha256 of the plaintext token.

    A vault token is 256 bits of ``secrets`` entropy, not a human password:
    a single sha256 is the right primitive (argon2 is for the admin login,
    where the secret is guessable).
    """
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def token_prefix(token: str) -> str:
    return token[:PREFIX_LEN]


def parse_bearer(header_values: list[str] | tuple[str, ...]) -> str | None:
    """Strict ``Authorization: Bearer dm0_…`` parse. None = malformed.

    Rejects: repeated Authorization headers, oversized values, any scheme but
    Bearer, and anything that is not exactly our token shape. Being strict
    here keeps the SQL lookup a single indexed equality on trusted-shape text.
    """
    if len(header_values) != 1:
        return None
    raw = header_values[0]
    if not raw or len(raw) > MAX_AUTH_HEADER_LEN:
        return None
    scheme, _, value = raw.partition(" ")
    if scheme.lower() != "bearer" or not value:
        return None
    token = value.strip()
    if len(token) != TOKEN_LEN or not _TOKEN_RE.match(token):
        return None
    return token


def probe(db_path: str | Path) -> str:
    """Readiness probe for /health: ok | missing | schema_incompatible | error."""
    path = Path(db_path)
    if not path.exists():
        return "missing"
    try:
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=2.0)
        try:
            version = conn.execute("PRAGMA user_version").fetchone()[0]
            if version > SCHEMA_VERSION:
                return "schema_incompatible"
            conn.execute("SELECT 1 FROM tokens LIMIT 1").fetchone()
        finally:
            conn.close()
    except Exception as exc:  # noqa: BLE001 - probe never raises
        logger.warning("vault probe failed for %s: %s", path, exc)
        return "error"
    return "ok"


class VaultStore:
    """Users, tokens and audit log over one SQLite file."""

    def __init__(self, db_path: str | Path, *, create: bool = True):
        self.db_path = str(db_path)
        if create:
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()
        self._restrict_permissions()

    def _restrict_permissions(self) -> None:
        """Owner-only on the vault and its WAL sidecars.

        systemd sets UMask=0077 for the service, but `bootstrap-admin` runs
        from an operator's shell with the usual 022 — which would leave token
        hashes, emails and the audit trail world-readable.
        """
        for suffix in ("", "-wal", "-shm"):
            path = Path(self.db_path + suffix)
            try:
                if path.exists() and (path.stat().st_mode & 0o077):
                    path.chmod(0o600)
            except OSError as exc:  # noqa: PERF203 - best effort, never fatal
                logger.warning("could not restrict permissions on %s: %s", path, exc)

    # ------------------------------------------------------------------
    # Connection / transaction plumbing
    # ------------------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=5.0, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=5000")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    @contextmanager
    def _tx(self) -> Iterator[sqlite3.Connection]:
        """One writer transaction; any exception rolls back mutation AND audit."""
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            try:
                yield conn
            except BaseException:
                conn.execute("ROLLBACK")
                raise
            conn.execute("COMMIT")
        finally:
            conn.close()

    @contextmanager
    def _read(self) -> Iterator[sqlite3.Connection]:
        conn = self._connect()
        try:
            yield conn
        finally:
            conn.close()

    def _init_schema(self) -> None:
        with self._tx() as conn:
            version = conn.execute("PRAGMA user_version").fetchone()[0]
            if version > SCHEMA_VERSION:
                raise SchemaIncompatible(
                    f"vault.db is schema v{version}, this build understands "
                    f"v{SCHEMA_VERSION} — upgrade the code or point "
                    f"MEM0_VAULT_DB_PATH at the matching file"
                )
            for stmt in _SCHEMA_STATEMENTS:
                conn.execute(stmt)
            for table, column, ddl in _COLUMN_MIGRATIONS:
                existing = {r["name"] for r in conn.execute(f"PRAGMA table_info({table})")}
                if column not in existing:
                    conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {ddl}")
            if version != SCHEMA_VERSION:
                # PRAGMA takes no parameters; the value is an int constant.
                conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")

    # ------------------------------------------------------------------
    # Audit (always inside the caller's transaction)
    # ------------------------------------------------------------------

    def _audit(
        self,
        conn: sqlite3.Connection,
        *,
        action: str,
        actor_id: int | None = None,
        actor_email: str = "",
        ip: str = "",
        subject_type: str = "",
        subject_id: str = "",
        success: bool = True,
        details: dict[str, Any] | None = None,
    ) -> None:
        conn.execute(
            "INSERT INTO audit_log (ts, actor_id, actor_email, ip, action,"
            " subject_type, subject_id, success, details)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                _iso(_utcnow()), actor_id, actor_email or "", ip or "", action,
                subject_type, str(subject_id), 1 if success else 0,
                json.dumps(details, ensure_ascii=False) if details else None,
            ),
        )

    def record_event(
        self,
        *,
        action: str,
        actor_id: int | None = None,
        actor_email: str = "",
        ip: str = "",
        subject_type: str = "",
        subject_id: str = "",
        success: bool = True,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Audit an event with no accompanying mutation (login ok/failed)."""
        with self._tx() as conn:
            self._audit(
                conn, action=action, actor_id=actor_id, actor_email=actor_email,
                ip=ip, subject_type=subject_type, subject_id=subject_id,
                success=success, details=details,
            )

    # ------------------------------------------------------------------
    # Users
    # ------------------------------------------------------------------

    def create_user(
        self,
        *,
        email: str,
        display_name: str = "",
        is_admin: bool = False,
        password_hash: str | None = None,
        mem0_user_id: str = "",
        actor_id: int | None = None,
        actor_email: str = "",
        ip: str = "",
    ) -> int:
        with self._tx() as conn:
            try:
                cur = conn.execute(
                    "INSERT INTO users (email, display_name, is_admin, password_hash,"
                    " mem0_user_id, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                    (email, display_name, 1 if is_admin else 0, password_hash,
                     mem0_user_id, _iso(_utcnow())),
                )
            except sqlite3.IntegrityError as exc:
                raise DuplicateEmail(email) from exc
            user_id = int(cur.lastrowid)
            self._audit(
                conn, action="user.create", actor_id=actor_id, actor_email=actor_email,
                ip=ip, subject_type="user", subject_id=str(user_id),
                details={"email": email, "is_admin": bool(is_admin)},
            )
            return user_id

    def get_user(self, user_id: int) -> dict[str, Any] | None:
        with self._read() as conn:
            row = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
            return dict(row) if row else None

    def get_user_by_email(self, email: str) -> dict[str, Any] | None:
        with self._read() as conn:
            row = conn.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()
            return dict(row) if row else None

    def list_users(self) -> list[dict[str, Any]]:
        """Users with token counts and last use, for the dashboard table."""
        now = _iso(_utcnow())
        with self._read() as conn:
            rows = conn.execute(
                "SELECT u.*,"
                " (SELECT COUNT(*) FROM tokens t WHERE t.user_id = u.id"
                "    AND t.revoked_at IS NULL"
                "    AND (t.expires_at IS NULL OR t.expires_at > ?)) AS active_tokens,"
                " (SELECT COUNT(*) FROM tokens t WHERE t.user_id = u.id"
                "    AND (t.revoked_at IS NOT NULL"
                "         OR (t.expires_at IS NOT NULL AND t.expires_at <= ?))) AS inactive_tokens,"
                " (SELECT MAX(t.last_used_at) FROM tokens t WHERE t.user_id = u.id) AS last_used_at"
                " FROM users u ORDER BY u.disabled_at IS NOT NULL, u.id",
                (now, now),
            ).fetchall()
            return [dict(r) for r in rows]

    def set_user_disabled(
        self,
        user_id: int,
        disabled: bool,
        *,
        actor_id: int | None = None,
        actor_email: str = "",
        ip: str = "",
    ) -> int:
        """Disable (revoking every token) or re-enable a user.

        Re-enabling does NOT resurrect tokens: revocation is final, the admin
        issues a new one. Returns how many tokens were revoked.
        """
        now = _iso(_utcnow())
        with self._tx() as conn:
            row = conn.execute("SELECT id FROM users WHERE id = ?", (user_id,)).fetchone()
            if row is None:
                raise NotFound(f"user {user_id}")
            revoked = 0
            if disabled:
                conn.execute("UPDATE users SET disabled_at = ? WHERE id = ?", (now, user_id))
                cur = conn.execute(
                    "UPDATE tokens SET revoked_at = ? WHERE user_id = ? AND revoked_at IS NULL",
                    (now, user_id),
                )
                revoked = cur.rowcount
            else:
                conn.execute("UPDATE users SET disabled_at = NULL WHERE id = ?", (user_id,))
            self._audit(
                conn, action="user.disable" if disabled else "user.enable",
                actor_id=actor_id, actor_email=actor_email, ip=ip,
                subject_type="user", subject_id=str(user_id),
                details={"tokens_revoked": revoked} if disabled else None,
            )
            return revoked

    def bump_session_epoch(self, user_id: int, *, reason: str = "logout") -> int:
        """Invalida TODA sessão já emitida para este usuário. Devolve a época nova."""
        with self._tx() as conn:
            conn.execute(
                "UPDATE users SET session_epoch = session_epoch + 1 WHERE id = ?", (user_id,)
            )
            row = conn.execute(
                "SELECT session_epoch FROM users WHERE id = ?", (user_id,)
            ).fetchone()
            if row is None:
                raise NotFound(f"user {user_id}")
            self._audit(
                conn, action="session.invalidate", actor_id=user_id,
                subject_type="user", subject_id=str(user_id), details={"reason": reason},
            )
            return int(row["session_epoch"])

    def set_password_hash(self, user_id: int, password_hash: str) -> None:
        with self._tx() as conn:
            # Trocar a senha derruba as sessões abertas — é o que qualquer um
            # espera de "mudei minha senha porque acho que vazou".
            conn.execute(
                "UPDATE users SET session_epoch = session_epoch + 1 WHERE id = ?", (user_id,)
            )
            cur = conn.execute(
                "UPDATE users SET password_hash = ? WHERE id = ?", (password_hash, user_id)
            )
            if cur.rowcount == 0:
                raise NotFound(f"user {user_id}")
            self._audit(
                conn, action="user.password", actor_id=user_id,
                subject_type="user", subject_id=str(user_id),
            )

    def count_admins(self) -> int:
        with self._read() as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS n FROM users WHERE is_admin = 1 AND disabled_at IS NULL"
            ).fetchone()
            return int(row["n"])

    # ------------------------------------------------------------------
    # Tokens
    # ------------------------------------------------------------------

    def create_token(
        self,
        *,
        user_id: int,
        token: str,
        label: str = "",
        expires_at: str | None = None,
        on_demand: bool = False,
        actor_id: int | None = None,
        actor_email: str = "",
        ip: str = "",
    ) -> dict[str, Any]:
        with self._tx() as conn:
            user = conn.execute(
                "SELECT id, disabled_at FROM users WHERE id = ?", (user_id,)
            ).fetchone()
            if user is None:
                raise NotFound(f"user {user_id}")
            if user["disabled_at"]:
                raise VaultError("cannot issue a token for a disabled user")
            token_id = self._insert_token(
                conn, user_id=user_id, token=token, label=label, expires_at=expires_at,
            )
            if on_demand:
                conn.execute("UPDATE tokens SET on_demand = 1 WHERE id = ?", (token_id,))
            self._audit(
                conn, action="token.create", actor_id=actor_id, actor_email=actor_email,
                ip=ip, subject_type="token", subject_id=str(token_id),
                details={"user_id": user_id, "prefix": token_prefix(token), "label": label},
            )
            return {"id": token_id, "prefix": token_prefix(token)}

    def _insert_token(
        self,
        conn: sqlite3.Connection,
        *,
        user_id: int,
        token: str,
        label: str,
        expires_at: str | None,
        renewed_from: int | None = None,
    ) -> int:
        try:
            cur = conn.execute(
                "INSERT INTO tokens (user_id, label, token_hash, prefix, created_at,"
                " expires_at, renewed_from) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (user_id, label, hash_token(token), token_prefix(token),
                 _iso(_utcnow()), expires_at, renewed_from),
            )
        except sqlite3.IntegrityError as exc:
            # Two different unique constraints, two different meanings:
            # UNIQUE(token_hash) is astronomically unlikely and retrying with a
            # fresh secret fixes it; UNIQUE(renewed_from) means a concurrent
            # rotation already won, and no amount of retrying will change that.
            if "renewed_from" in str(exc):
                raise SuccessorExists(str(exc)) from exc
            raise TokenCollision(str(exc)) from exc
        return int(cur.lastrowid)

    def rotate_token(
        self,
        token_id: int,
        *,
        new_token: str,
        grace_seconds: int,
        actor_id: int | None = None,
        actor_email: str = "",
        ip: str = "",
    ) -> dict[str, Any]:
        """Issue a successor and put the old token on a migration clock.

        The old token is NOT revoked — it expires after ``grace_seconds`` so a
        client can be reconfigured without an outage. If it already had an
        earlier expiry, that stands (rotation never extends a token's life).
        """
        now = _utcnow()
        grace_until = _iso(now + timedelta(seconds=max(grace_seconds, 0)))
        with self._tx() as conn:
            old = conn.execute("SELECT * FROM tokens WHERE id = ?", (token_id,)).fetchone()
            if old is None:
                raise NotFound(f"token {token_id}")
            if old["revoked_at"]:
                raise VaultError("cannot rotate a revoked token")
            new_id = self._insert_token(
                conn, user_id=old["user_id"], token=new_token, label=old["label"],
                expires_at=old["expires_at"], renewed_from=token_id,
            )
            if old["expires_at"] is None or old["expires_at"] > grace_until:
                conn.execute(
                    "UPDATE tokens SET expires_at = ? WHERE id = ?", (grace_until, token_id)
                )
            self._audit(
                conn, action="token.rotate", actor_id=actor_id, actor_email=actor_email,
                ip=ip, subject_type="token", subject_id=str(token_id),
                details={
                    "successor_id": new_id, "successor_prefix": token_prefix(new_token),
                    "old_prefix": old["prefix"], "grace_until": grace_until,
                },
            )
            return {"id": new_id, "prefix": token_prefix(new_token), "grace_until": grace_until}

    def revoke_token(
        self,
        token_id: int,
        *,
        actor_id: int | None = None,
        actor_email: str = "",
        ip: str = "",
    ) -> None:
        with self._tx() as conn:
            row = conn.execute("SELECT * FROM tokens WHERE id = ?", (token_id,)).fetchone()
            if row is None:
                raise NotFound(f"token {token_id}")
            if row["revoked_at"] is None:
                conn.execute(
                    "UPDATE tokens SET revoked_at = ? WHERE id = ?", (_iso(_utcnow()), token_id)
                )
            self._audit(
                conn, action="token.revoke", actor_id=actor_id, actor_email=actor_email,
                ip=ip, subject_type="token", subject_id=str(token_id),
                details={"user_id": row["user_id"], "prefix": row["prefix"]},
            )

    def list_tokens(self, user_id: int) -> list[dict[str, Any]]:
        with self._read() as conn:
            rows = conn.execute(
                "SELECT * FROM tokens WHERE user_id = ? ORDER BY revoked_at IS NOT NULL, id DESC",
                (user_id,),
            ).fetchall()
            return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Verification (hot path — the MCP middleware calls this per request)
    # ------------------------------------------------------------------

    def verify_token(self, token: str, *, touch: bool = True) -> dict[str, Any]:
        """Resolve a plaintext token to a principal.

        Returns ``{"status": ..., ...principal}``. Status order is
        unknown → revoked → expired → user_disabled → ok, so the most
        specific reason is what gets logged.
        """
        if not token or len(token) != TOKEN_LEN or not _TOKEN_RE.match(token):
            return {"status": MALFORMED}

        with self._read() as conn:
            row = conn.execute(
                "SELECT t.id AS token_id, t.user_id, t.prefix, t.label, t.expires_at,"
                " t.revoked_at, t.last_used_at, u.email, u.mem0_user_id, u.is_admin,"
                " u.disabled_at"
                " FROM tokens t JOIN users u ON u.id = t.user_id"
                " WHERE t.token_hash = ?",
                (hash_token(token),),
            ).fetchone()

            if row is None:
                return {"status": UNKNOWN}

            now = _utcnow()
            if row["revoked_at"] is not None:
                status = REVOKED
            elif row["expires_at"] is not None and row["expires_at"] <= _iso(now):
                status = EXPIRED
            elif row["disabled_at"] is not None:
                status = USER_DISABLED
            else:
                status = OK

            principal = {
                "status": status,
                "token_id": row["token_id"],
                "prefix": row["prefix"],
                "label": row["label"],
                "user_id": row["user_id"],
                "email": row["email"],
                "mem0_user_id": row["mem0_user_id"] or "",
                "is_admin": bool(row["is_admin"]),
            }

        if status == OK and touch:
            self._touch(row["token_id"], row["last_used_at"], now)
        return principal

    def _touch(self, token_id: int, last_used_at: str | None, now: datetime) -> None:
        """Best-effort last_used_at, at most once a minute per token.

        Conditional UPDATE (not read-then-write) so concurrent requests can't
        clobber each other, and a locked database never fails a request that
        is otherwise authorized.
        """
        cutoff = _iso(now - timedelta(seconds=60))
        if last_used_at is not None and last_used_at > cutoff:
            return
        try:
            with self._tx() as conn:
                conn.execute(
                    "UPDATE tokens SET last_used_at = ?, use_count = use_count + 1"
                    " WHERE id = ? AND (last_used_at IS NULL OR last_used_at < ?)",
                    (_iso(now), token_id, cutoff),
                )
        except sqlite3.OperationalError as exc:  # database is locked
            logger.debug("vault touch skipped for token %s: %s", token_id, exc)

    # ------------------------------------------------------------------
    # Rollout evidence
    # ------------------------------------------------------------------

    def record_denials(self, counts: dict[str, int], *, path: str = "", client: str = "") -> None:
        """Accumulate unauthorized attempts into the current time bucket.

        Bucketed and called from a caller-side throttle, so an attacker
        hammering 401s costs one UPDATE per bucket per reason — not one write
        per request.
        """
        if not counts:
            return
        now = _utcnow()
        bucket = _iso(now.replace(
            minute=(now.minute // (DENIAL_BUCKET_S // 60)) * (DENIAL_BUCKET_S // 60),
            second=0, microsecond=0,
        ))
        with self._tx() as conn:
            for reason, count in counts.items():
                if count <= 0:
                    continue
                conn.execute(
                    "INSERT INTO auth_denials (bucket_start, reason, count, last_seen_at,"
                    " last_path, last_client) VALUES (?, ?, ?, ?, ?, ?)"
                    " ON CONFLICT(bucket_start, reason) DO UPDATE SET"
                    " count = count + excluded.count, last_seen_at = excluded.last_seen_at,"
                    " last_path = excluded.last_path, last_client = excluded.last_client",
                    (bucket, reason, count, _iso(now), path[:200], client[:64]),
                )

    def denials_since(self, cutoff_iso: str) -> list[dict[str, Any]]:
        with self._read() as conn:
            rows = conn.execute(
                "SELECT reason, SUM(count) AS count, MAX(last_seen_at) AS last_seen_at,"
                " MAX(last_path) AS last_path, MAX(last_client) AS last_client"
                " FROM auth_denials WHERE bucket_start >= ?"
                " GROUP BY reason ORDER BY count DESC",
                (cutoff_iso,),
            ).fetchall()
            return [dict(r) for r in rows]

    def promotion_readiness(
        self, *, window_hours: int = 72, denial_window_hours: float = 1.0
    ) -> dict[str, Any]:
        """Can MEM0_REQUIRE_AUTH go from shadow to on?

        The inventory is not a separate list to keep in sync: every ACTIVE
        token IS an expected client. A token that never authorized in the
        window is exactly the client that would start taking 401s.

        Denials use a SHORTER window on purpose. The first version counted
        them over the whole 72 h and stayed NOT READY for three days because
        of the 401s produced by the migration it had just guided — punishing
        the transition instead of measuring the current state. What blocks a
        promotion is somebody being denied NOW; older denials are history and
        are reported as context, not as a veto.
        """
        now = _utcnow()
        cutoff = _iso(now - timedelta(hours=window_hours))
        denial_cutoff = _iso(now - timedelta(hours=denial_window_hours))
        with self._read() as conn:
            rows = conn.execute(
                "SELECT t.id, t.prefix, t.label, t.last_used_at, t.use_count,"
                " t.on_demand, u.email"
                " FROM tokens t JOIN users u ON u.id = t.user_id"
                " WHERE t.revoked_at IS NULL AND (t.expires_at IS NULL OR t.expires_at > ?)"
                " AND u.disabled_at IS NULL ORDER BY t.id",
                (_iso(now),),
            ).fetchall()
        every = [dict(r) for r in rows]
        # Clientes por demanda são listados, nunca vetam: eles não são
        # "esperados agora", são "esperados quando alguém os rodar".
        on_demand = [t for t in every if t["on_demand"]]
        tokens = [t for t in every if not t["on_demand"]]
        seen = [t for t in tokens if (t["last_used_at"] or "") > cutoff]
        silent = [t for t in tokens if (t["last_used_at"] or "") <= cutoff]
        denials = self.denials_since(denial_cutoff)
        historic = self.denials_since(cutoff)
        return {
            "window_hours": window_hours,
            "denial_window_hours": denial_window_hours,
            "expected": len(tokens),
            "seen": seen,
            "silent": silent,
            "on_demand": on_demand,
            "denials": denials,
            "denial_total": sum(d["count"] for d in denials),
            "denial_historic_total": sum(d["count"] for d in historic),
            # Silence is not evidence: a token nobody used is indistinguishable
            # from a client that would break, so it blocks promotion.
            "ready": bool(tokens) and not silent and not denials,
        }

    # ------------------------------------------------------------------
    # Read models for the UI
    # ------------------------------------------------------------------

    def list_audit(self, *, limit: int = 50, offset: int = 0) -> list[dict[str, Any]]:
        with self._read() as conn:
            rows = conn.execute(
                "SELECT * FROM audit_log ORDER BY id DESC LIMIT ? OFFSET ?",
                (max(1, min(limit, 500)), max(0, offset)),
            ).fetchall()
            return [dict(r) for r in rows]

    def count_audit(self) -> int:
        with self._read() as conn:
            return int(conn.execute("SELECT COUNT(*) AS n FROM audit_log").fetchone()["n"])

    def count_tokens_used_since(self, cutoff_iso: str) -> int:
        """Active tokens that authorized at least one request since ``cutoff``.

        This is the number the shadow->on promotion is read against, so it
        counts TOKENS (what a client holds), not users.
        """
        with self._read() as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS n FROM tokens"
                " WHERE last_used_at IS NOT NULL AND last_used_at > ?"
                " AND revoked_at IS NULL",
                (cutoff_iso,),
            ).fetchone()
            return int(row["n"])

    def stats(self) -> dict[str, int]:
        now = _iso(_utcnow())
        with self._read() as conn:
            row = conn.execute(
                "SELECT"
                " (SELECT COUNT(*) FROM users) AS users_total,"
                " (SELECT COUNT(*) FROM users WHERE disabled_at IS NULL) AS users_active,"
                " (SELECT COUNT(*) FROM tokens) AS tokens_total,"
                " (SELECT COUNT(*) FROM tokens WHERE revoked_at IS NULL"
                "    AND (expires_at IS NULL OR expires_at > ?)) AS tokens_active",
                (now,),
            ).fetchone()
            return {k: int(row[k]) for k in row.keys()}
