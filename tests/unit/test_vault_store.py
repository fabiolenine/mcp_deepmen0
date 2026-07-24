"""Vault storage: schema lifecycle, atomic mutations, token verification."""

import sqlite3
import secrets
import threading
from datetime import datetime, timedelta, timezone

import pytest

from mem0_mcp_selfhosted.vault import store as vs


def _token() -> str:
    return vs.TOKEN_PREFIX + secrets.token_urlsafe(32)


def _iso(dt: datetime) -> str:
    return dt.isoformat()


@pytest.fixture
def store(tmp_path):
    return vs.VaultStore(tmp_path / "vault.db")


@pytest.fixture
def user(store):
    return store.create_user(email="admin@example.com", display_name="Admin", is_admin=True)


# ---------------------------------------------------------------- schema


class TestSchema:
    def test_creates_file_and_sets_version(self, tmp_path):
        path = tmp_path / "nested" / "vault.db"
        vs.VaultStore(path)
        conn = sqlite3.connect(path)
        assert conn.execute("PRAGMA user_version").fetchone()[0] == vs.SCHEMA_VERSION
        conn.close()

    def test_reopen_is_idempotent(self, tmp_path):
        path = tmp_path / "vault.db"
        first = vs.VaultStore(path)
        uid = first.create_user(email="a@b.com")
        second = vs.VaultStore(path)
        assert second.get_user(uid)["email"] == "a@b.com"

    def test_two_stores_share_one_file(self, tmp_path):
        """Both processes (MCP + UI) open the same file concurrently."""
        path = tmp_path / "vault.db"
        writer, reader = vs.VaultStore(path), vs.VaultStore(path)
        uid = writer.create_user(email="x@y.com")
        tok = _token()
        writer.create_token(user_id=uid, token=tok)
        assert reader.verify_token(tok)["status"] == vs.OK

    def test_future_schema_is_refused(self, tmp_path):
        path = tmp_path / "vault.db"
        conn = sqlite3.connect(path)
        conn.execute(f"PRAGMA user_version = {vs.SCHEMA_VERSION + 1}")
        conn.commit()
        conn.close()
        with pytest.raises(vs.SchemaIncompatible):
            vs.VaultStore(path)

    def test_probe_states(self, tmp_path):
        assert vs.probe(tmp_path / "absent.db") == "missing"
        path = tmp_path / "vault.db"
        vs.VaultStore(path)
        assert vs.probe(path) == "ok"

        future = tmp_path / "future.db"
        conn = sqlite3.connect(future)
        conn.execute(f"PRAGMA user_version = {vs.SCHEMA_VERSION + 9}")
        conn.commit()
        conn.close()
        assert vs.probe(future) == "schema_incompatible"

        garbage = tmp_path / "garbage.db"
        garbage.write_bytes(b"definitely not sqlite")
        assert vs.probe(garbage) == "error"


# ---------------------------------------------------------------- parsing


class TestParseBearer:
    def test_accepts_well_formed(self):
        tok = _token()
        assert vs.parse_bearer([f"Bearer {tok}"]) == tok
        assert vs.parse_bearer([f"bearer {tok}"]) == tok  # RFC: scheme is case-insensitive

    @pytest.mark.parametrize(
        "headers",
        [
            [],                                          # absent
            ["Bearer a", "Bearer b"],                    # repeated header
            ["Basic " + "x" * 43],                       # wrong scheme
            ["Bearer"],                                  # no value
            ["Bearer dm0_short"],                        # wrong length
            ["Bearer xyz_" + "a" * 43],                  # wrong prefix
            ["Bearer dm0_" + "a" * 42 + "!"],            # charset
            ["Bearer " + "a" * (vs.MAX_AUTH_HEADER_LEN + 1)],  # oversized
        ],
    )
    def test_rejects(self, headers):
        assert vs.parse_bearer(headers) is None


# ---------------------------------------------------------------- verify


class TestVerifyToken:
    def test_ok(self, store, user):
        tok = _token()
        store.create_token(user_id=user, token=tok, label="laptop")
        res = store.verify_token(tok)
        assert res["status"] == vs.OK
        assert res["email"] == "admin@example.com"
        assert res["prefix"] == tok[:12]
        assert res["mem0_user_id"] == ""

    def test_malformed_never_hits_the_database(self, store):
        assert store.verify_token("nope")["status"] == vs.MALFORMED

    def test_unknown(self, store, user):
        store.create_token(user_id=user, token=_token())
        assert store.verify_token(_token())["status"] == vs.UNKNOWN

    def test_revoked(self, store, user):
        tok = _token()
        tid = store.create_token(user_id=user, token=tok)["id"]
        store.revoke_token(tid)
        assert store.verify_token(tok)["status"] == vs.REVOKED

    def test_expired(self, store, user):
        tok = _token()
        past = _iso(datetime.now(timezone.utc) - timedelta(seconds=1))
        store.create_token(user_id=user, token=tok, expires_at=past)
        assert store.verify_token(tok)["status"] == vs.EXPIRED

    def test_user_disabled(self, store, user):
        tok = _token()
        store.create_token(user_id=user, token=tok)
        store.set_user_disabled(user, True)
        # disabling revokes tokens, so the more specific reason wins
        assert store.verify_token(tok)["status"] == vs.REVOKED

    def test_disabled_user_with_untouched_token(self, store, user):
        """A token issued before the schema-level revoke still reports the user."""
        tok = _token()
        store.create_token(user_id=user, token=tok)
        with store._tx() as conn:  # disable WITHOUT the cascade, to isolate the check
            conn.execute("UPDATE users SET disabled_at = ? WHERE id = ?",
                         (_iso(datetime.now(timezone.utc)), user))
        assert store.verify_token(tok)["status"] == vs.USER_DISABLED

    def test_carries_mem0_user_id_binding(self, store):
        uid = store.create_user(email="bound@example.com", mem0_user_id="alice")
        tok = _token()
        store.create_token(user_id=uid, token=tok)
        assert store.verify_token(tok)["mem0_user_id"] == "alice"


class TestTouch:
    def test_first_use_records_timestamp(self, store, user):
        tok = _token()
        tid = store.create_token(user_id=user, token=tok)["id"]
        store.verify_token(tok)
        assert store.list_tokens(user)[0]["last_used_at"] is not None
        assert tid == store.list_tokens(user)[0]["id"]

    def test_throttled_to_once_a_minute(self, store, user):
        tok = _token()
        store.verify_token_calls = 0
        store.create_token(user_id=user, token=tok)
        store.verify_token(tok)
        first = store.list_tokens(user)[0]["last_used_at"]
        store.verify_token(tok)
        assert store.list_tokens(user)[0]["last_used_at"] == first

    def test_stale_timestamp_is_refreshed(self, store, user):
        tok = _token()
        tid = store.create_token(user_id=user, token=tok)["id"]
        old = _iso(datetime.now(timezone.utc) - timedelta(hours=2))
        with store._tx() as conn:
            conn.execute("UPDATE tokens SET last_used_at = ? WHERE id = ?", (old, tid))
        store.verify_token(tok)
        assert store.list_tokens(user)[0]["last_used_at"] > old

    def test_locked_database_never_fails_the_request(self, store, user, monkeypatch):
        tok = _token()
        store.create_token(user_id=user, token=tok)

        def boom(*_a, **_kw):
            raise sqlite3.OperationalError("database is locked")

        monkeypatch.setattr(store, "_tx", boom)
        assert store.verify_token(tok)["status"] == vs.OK  # touch failure swallowed

    def test_verify_without_touch(self, store, user):
        tok = _token()
        store.create_token(user_id=user, token=tok)
        store.verify_token(tok, touch=False)
        assert store.list_tokens(user)[0]["last_used_at"] is None


# ---------------------------------------------------------------- atomicity


class TestAtomicity:
    def test_audit_failure_rolls_back_the_mutation(self, store, user, monkeypatch):
        """Injected failure between mutation and audit: neither survives."""
        tok = _token()
        tid = store.create_token(user_id=user, token=tok)["id"]
        audits_before = len(store.list_audit(limit=500))

        def exploding_audit(*_a, **_kw):
            raise RuntimeError("audit writer down")

        monkeypatch.setattr(store, "_audit", exploding_audit)
        with pytest.raises(RuntimeError):
            store.revoke_token(tid)
        monkeypatch.undo()

        assert store.list_tokens(user)[0]["revoked_at"] is None
        assert len(store.list_audit(limit=500)) == audits_before
        assert store.verify_token(tok)["status"] == vs.OK

    def test_every_mutation_writes_exactly_one_audit_row(self, store, user):
        tok = _token()
        tid = store.create_token(user_id=user, token=tok)["id"]
        store.rotate_token(tid, new_token=_token(), grace_seconds=60)
        store.revoke_token(tid)
        store.set_user_disabled(user, True)
        actions = [a["action"] for a in store.list_audit(limit=100)]
        assert actions == [
            "user.disable", "token.revoke", "token.rotate", "token.create", "user.create",
        ]

    def test_audit_never_stores_plaintext(self, store, user):
        tok = _token()
        store.create_token(user_id=user, token=tok, label="laptop")
        blob = str(store.list_audit(limit=10))
        assert tok not in blob
        assert tok[:12] in blob  # the prefix, and only the prefix


# ---------------------------------------------------------------- tokens


class TestTokens:
    def test_hash_collision_raises_for_the_caller_to_regenerate(self, store, user):
        tok = _token()
        store.create_token(user_id=user, token=tok)
        with pytest.raises(vs.TokenCollision):
            store.create_token(user_id=user, token=tok)

    def test_disabled_user_cannot_receive_tokens(self, store, user):
        store.set_user_disabled(user, True)
        with pytest.raises(vs.VaultError):
            store.create_token(user_id=user, token=_token())

    def test_duplicate_email_refused(self, store, user):
        with pytest.raises(vs.DuplicateEmail):
            store.create_user(email="admin@example.com")

    def test_unknown_user_or_token(self, store):
        with pytest.raises(vs.NotFound):
            store.create_token(user_id=999, token=_token())
        with pytest.raises(vs.NotFound):
            store.revoke_token(999)
        with pytest.raises(vs.NotFound):
            store.set_user_disabled(999, True)


class TestRotation:
    def test_grace_window_keeps_the_old_token_alive(self, store, user):
        old_tok, new_tok = _token(), _token()
        tid = store.create_token(user_id=user, token=old_tok)["id"]
        res = store.rotate_token(tid, new_token=new_tok, grace_seconds=86400)

        assert store.verify_token(old_tok)["status"] == vs.OK  # still valid
        assert store.verify_token(new_tok)["status"] == vs.OK
        assert res["grace_until"] > _iso(datetime.now(timezone.utc))

    def test_rotation_never_extends_an_earlier_expiry(self, store, user):
        tok = _token()
        soon = _iso(datetime.now(timezone.utc) + timedelta(seconds=30))
        tid = store.create_token(user_id=user, token=tok, expires_at=soon)["id"]
        store.rotate_token(tid, new_token=_token(), grace_seconds=86400)
        assert store.list_tokens(user)[-1]["expires_at"] == soon

    def test_only_one_successor_survives_concurrent_rotation(self, store, user):
        tid = store.create_token(user_id=user, token=_token())["id"]
        errors, wins = [], []
        barrier = threading.Barrier(4)

        def rotate():
            barrier.wait()
            try:
                wins.append(store.rotate_token(tid, new_token=_token(), grace_seconds=60))
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [threading.Thread(target=rotate) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(wins) == 1, f"expected exactly one successor, got {len(wins)}"
        assert len(errors) == 3
        # SuccessorExists, not TokenCollision: retrying with a fresh secret
        # cannot win, so the web layer must not loop on it.
        assert all(isinstance(e, vs.SuccessorExists) for e in errors)
        successors = [t for t in store.list_tokens(user) if t["renewed_from"] == tid]
        assert len(successors) == 1

    def test_revoked_token_cannot_be_rotated(self, store, user):
        tid = store.create_token(user_id=user, token=_token())["id"]
        store.revoke_token(tid)
        with pytest.raises(vs.VaultError):
            store.rotate_token(tid, new_token=_token(), grace_seconds=60)

    def test_revoke_is_idempotent(self, store, user):
        tok = _token()
        tid = store.create_token(user_id=user, token=tok)["id"]
        store.revoke_token(tid)
        first = store.list_tokens(user)[0]["revoked_at"]
        store.revoke_token(tid)
        assert store.list_tokens(user)[0]["revoked_at"] == first


class TestUsers:
    def test_disable_revokes_every_token_and_reenable_does_not_restore(self, store, user):
        toks = [_token() for _ in range(3)]
        for t in toks:
            store.create_token(user_id=user, token=t)

        assert store.set_user_disabled(user, True) == 3
        assert all(store.verify_token(t)["status"] == vs.REVOKED for t in toks)

        store.set_user_disabled(user, False)
        assert all(store.verify_token(t)["status"] == vs.REVOKED for t in toks)

    def test_list_users_counts_and_stats(self, store, user):
        active, expired = _token(), _token()
        store.create_token(user_id=user, token=active)
        store.create_token(
            user_id=user, token=expired,
            expires_at=_iso(datetime.now(timezone.utc) - timedelta(days=1)),
        )
        row = store.list_users()[0]
        assert row["active_tokens"] == 1
        assert row["inactive_tokens"] == 1
        assert store.stats() == {
            "users_total": 1, "users_active": 1, "tokens_total": 2, "tokens_active": 1,
        }

    def test_audit_pagination(self, store, user):
        for _ in range(5):
            store.create_token(user_id=user, token=_token())
        assert store.count_audit() == 6  # 5 tokens + user.create
        page = store.list_audit(limit=2, offset=2)
        assert len(page) == 2
        assert page[0]["id"] > page[1]["id"]  # newest first


class TestPromotionEvidence:
    """The number the shadow->on decision is read against counts TOKENS."""

    def test_counts_tokens_not_users(self, store, user):
        from datetime import datetime, timedelta, timezone

        recent = _iso(datetime.now(timezone.utc) - timedelta(minutes=5))
        stale = _iso(datetime.now(timezone.utc) - timedelta(days=3))
        cutoff = _iso(datetime.now(timezone.utc) - timedelta(hours=24))

        ids = [store.create_token(user_id=user, token=_token())["id"] for _ in range(3)]
        with store._tx() as conn:
            conn.execute("UPDATE tokens SET last_used_at = ? WHERE id IN (?, ?)",
                         (recent, ids[0], ids[1]))
            conn.execute("UPDATE tokens SET last_used_at = ? WHERE id = ?", (stale, ids[2]))

        # two tokens of the SAME user were seen: the count must be 2, not 1
        assert store.count_tokens_used_since(cutoff) == 2

    def test_revoked_tokens_do_not_count_as_evidence(self, store, user):
        from datetime import datetime, timedelta, timezone

        recent = _iso(datetime.now(timezone.utc) - timedelta(minutes=5))
        cutoff = _iso(datetime.now(timezone.utc) - timedelta(hours=24))
        tid = store.create_token(user_id=user, token=_token())["id"]
        with store._tx() as conn:
            conn.execute("UPDATE tokens SET last_used_at = ? WHERE id = ?", (recent, tid))
        assert store.count_tokens_used_since(cutoff) == 1
        store.revoke_token(tid)
        assert store.count_tokens_used_since(cutoff) == 0
