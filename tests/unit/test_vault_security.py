"""Token generation, password hashing, input validation, CSRF."""

import re

import pytest

from mem0_mcp_selfhosted.vault import security as sec
from mem0_mcp_selfhosted.vault import store as vs


class TestTokenGeneration:
    def test_shape_matches_what_the_gate_accepts(self):
        token = sec.generate_token()
        assert vs.parse_bearer([f"Bearer {token}"]) == token
        assert token.startswith(vs.TOKEN_PREFIX)
        assert len(token) == vs.TOKEN_LEN

    def test_entropy_no_repeats(self):
        assert len({sec.generate_token() for _ in range(500)}) == 500

    def test_prefix_is_twelve_chars(self):
        token = sec.generate_token()
        assert len(vs.token_prefix(token)) == 12
        assert vs.token_prefix(token) == token[:12]


class TestEmailValidation:
    @pytest.mark.parametrize("raw,expected", [
        ("Admin@Example.COM", "admin@example.com"),
        ("  dev@acme.com.br  ", "dev@acme.com.br"),
        ("first.last+tag@sub.domain.io", "first.last+tag@sub.domain.io"),
        ("openwebui@interno.lan", "openwebui@interno.lan"),
        ("a1_b@x-y.co", "a1_b@x-y.co"),
    ])
    def test_accepts_and_normalizes(self, raw, expected):
        assert sec.normalize_email(raw) == expected

    @pytest.mark.parametrize("raw", [
        "", "   ", "no-at-sign", "@nolocal.com", "user@", "user@nodot",
        "user..double@x.com", "user@x..com", "user@-x.com", ".start@x.com",
        "user name@x.com", "user@x.com.", "üser@x.com", "user@exämple.com",
        "a" * 65 + "@x.com", "a" * 250 + "@example.com",
        "user@x.com\nBcc: evil@x.com", "<script>@x.com",
    ])
    def test_rejects(self, raw):
        with pytest.raises(sec.ValidationError):
            sec.normalize_email(raw)

    def test_error_codes_are_translatable(self):
        from mem0_mcp_selfhosted.vault import i18n

        with pytest.raises(sec.ValidationError) as exc:
            sec.normalize_email("nope")
        assert i18n.error_message("pt", str(exc.value)) == "E-mail inválido — verifique o formato."
        assert i18n.error_message("en", str(exc.value)) == "Invalid email — check the format."


class TestNameAndLabel:
    def test_name_is_trimmed(self):
        assert sec.normalize_name("  Ana Souza  ") == "Ana Souza"

    @pytest.mark.parametrize("raw", ["", "   ", "x" * 201])
    def test_name_rejects(self, raw):
        with pytest.raises(sec.ValidationError):
            sec.normalize_name(raw)

    @pytest.mark.parametrize("raw,expected", [
        ("claude-code-local", "claude-code-local"),
        ("  open-webui  ", "open-webui"),
        ("", "token"),
        ("macbook mcp-remote", "macbook mcp-remote"),
    ])
    def test_label_accepts(self, raw, expected):
        assert sec.normalize_label(raw) == expected

    @pytest.mark.parametrize("raw", ["x" * 61, "bad\nlabel", "a<b>", "drop;table"])
    def test_label_rejects(self, raw):
        with pytest.raises(sec.ValidationError):
            sec.normalize_label(raw)


class TestPasswords:
    def test_roundtrip(self):
        h = sec.hash_password("correct horse battery")
        assert h.startswith("$argon2id$")
        assert sec.verify_password(h, "correct horse battery")
        assert not sec.verify_password(h, "wrong")

    def test_short_password_refused(self):
        with pytest.raises(sec.ValidationError):
            sec.hash_password("short")

    def test_missing_hash_is_false_not_a_crash(self):
        assert not sec.verify_password(None, "anything")
        assert not sec.verify_password("", "anything")

    def test_malformed_hash_is_false_not_a_crash(self):
        assert not sec.verify_password("not-a-hash", "anything")

    def test_hashes_are_salted(self):
        assert sec.hash_password("same password") != sec.hash_password("same password")


class TestCsrf:
    def test_match(self):
        token = sec.new_csrf_token()
        assert sec.csrf_ok(token, token)

    @pytest.mark.parametrize("session,form", [
        (None, "x"), ("x", None), ("", ""), ("a", "b"), ("abc", "abcd"),
    ])
    def test_mismatch(self, session, form):
        assert not sec.csrf_ok(session, form)


class TestPresentation:
    @pytest.mark.parametrize("name,expected", [
        ("Ana Souza", "AS"), ("Open WebUI (serviço)", "OW"), ("carlos", "C"), ("", "?"),
    ])
    def test_initials(self, name, expected):
        assert sec.initials(name) == expected

    def test_avatar_hue_is_stable_and_bounded(self):
        assert sec.avatar_hue("ana@x.com") == sec.avatar_hue("ana@x.com")
        assert 0 <= sec.avatar_hue("ana@x.com") < 360
        assert sec.avatar_hue("ana@x.com") != sec.avatar_hue("carlos@x.com")


def test_no_plaintext_token_pattern_in_repository_logs():
    """Guard the format itself: 'dm0_' + 43 urlsafe chars is what we grep for."""
    token = sec.generate_token()
    assert re.match(r"^dm0_[A-Za-z0-9_-]{43}$", token)
