"""Secrets, hashing, validation and CSRF for the vault UI.

Two different secrets, two different primitives:

- **API tokens** are 256 bits from ``secrets`` — unguessable by construction,
  so a plain sha256 lookup (in ``store``) is correct and fast.
- **The admin password** is human-chosen, so it gets argon2id. The import is
  lazy: the MCP gate must keep working without the ``[vault]`` extra.
"""

from __future__ import annotations

import hmac
import re
import secrets

from mem0_mcp_selfhosted.vault.store import TOKEN_BODY_LEN, TOKEN_PREFIX

MAX_EMAIL_LEN = 254
MAX_EMAIL_LOCAL_LEN = 64
MAX_FORM_FIELD_LEN = 200

# Deliberately ASCII-strict: this validates an identifier the admin types for
# a machine account, not the full RFC 5322 grammar. IDN/unicode is future work
# and would need normalization rules to stay comparable as a UNIQUE key.
# Stricter than the mockup's client-side regex on one point: a domain label may
# not start or end with a hyphen (the prototype accepted "user@-x.com").
_DOMAIN_LABEL = r"[a-zA-Z0-9](?:[a-zA-Z0-9-]*[a-zA-Z0-9])?"
_EMAIL_RE = re.compile(
    rf"^[a-zA-Z0-9](\.?[a-zA-Z0-9_+-])*@{_DOMAIN_LABEL}(\.{_DOMAIN_LABEL})+$"
)

_LABEL_RE = re.compile(r"^[\w .@:+-]{1,60}$", re.UNICODE)


class ValidationError(ValueError):
    """Rejected user input (message is safe to show)."""


def generate_token() -> str:
    """A fresh API token: ``dm0_`` + 43 urlsafe chars (256 bits)."""
    token = TOKEN_PREFIX + secrets.token_urlsafe(32)
    assert len(token) == len(TOKEN_PREFIX) + TOKEN_BODY_LEN  # noqa: S101 - shape invariant
    return token


def normalize_email(raw: str) -> str:
    """Lowercase + validate. Raises ValidationError with a UI-safe message."""
    email = (raw or "").strip().lower()
    if not email:
        raise ValidationError("email_required")
    if len(email) > MAX_EMAIL_LEN or ".." in email:
        raise ValidationError("email_invalid")
    local, _, domain = email.partition("@")
    if not domain or len(local) > MAX_EMAIL_LOCAL_LEN or email.endswith("."):
        raise ValidationError("email_invalid")
    if not _EMAIL_RE.match(email):
        raise ValidationError("email_invalid")
    return email


def normalize_name(raw: str) -> str:
    name = (raw or "").strip()
    if not name:
        raise ValidationError("name_required")
    if len(name) > MAX_FORM_FIELD_LEN:
        raise ValidationError("name_too_long")
    return name


def normalize_label(raw: str, *, default: str = "token") -> str:
    """Token label: short, printable, no control characters."""
    label = (raw or "").strip() or default
    if len(label) > 60:
        raise ValidationError("label_too_long")
    if not _LABEL_RE.match(label):
        raise ValidationError("label_invalid")
    return label


def initials(name: str) -> str:
    """Two-letter avatar initials, as in the design."""
    words = [w for w in re.split(r"[\s(]+", name or "") if w]
    return "".join(w[0].upper() for w in words[:2]) or "?"


def avatar_hue(seed: str) -> int:
    """Stable per-user hue (the mock varies avatar color by user)."""
    return sum(ord(c) * (i + 7) for i, c in enumerate(seed or "?")) % 360


# ---------------------------------------------------------------- passwords


def hash_password(password: str) -> str:
    from argon2 import PasswordHasher

    if not password or len(password) < 8:
        raise ValidationError("password_too_short")
    return PasswordHasher().hash(password)


def verify_password(stored_hash: str | None, password: str) -> bool:
    """Constant-ish time check; a missing hash still costs a verification."""
    from argon2 import PasswordHasher
    from argon2.exceptions import VerificationError, VerifyMismatchError

    hasher = PasswordHasher()
    if not stored_hash:
        # Don't leak "no such user" through timing: hash the input anyway.
        hasher.hash(password or "x")
        return False
    try:
        return hasher.verify(stored_hash, password or "")
    except (VerifyMismatchError, VerificationError):
        return False
    except Exception:  # noqa: BLE001 - malformed hash in the database
        return False


def needs_rehash(stored_hash: str) -> bool:
    from argon2 import PasswordHasher

    try:
        return PasswordHasher().check_needs_rehash(stored_hash)
    except Exception:  # noqa: BLE001
        return False


# ---------------------------------------------------------------- CSRF


def new_csrf_token() -> str:
    return secrets.token_urlsafe(24)


def csrf_ok(session_token: str | None, form_token: str | None) -> bool:
    if not session_token or not form_token:
        return False
    return hmac.compare_digest(session_token, form_token)
