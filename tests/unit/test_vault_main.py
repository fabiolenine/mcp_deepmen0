"""The vault entry point: bootstrap-admin and the packaged assets."""

import pytest

from mem0_mcp_selfhosted.vault import main as vault_main
from mem0_mcp_selfhosted.vault import security as sec
from mem0_mcp_selfhosted.vault import store as vs

PASSWORD = "uma senha longa o suficiente"


@pytest.fixture(autouse=True)
def _isolated_db(tmp_path, monkeypatch):
    monkeypatch.setenv("MEM0_VAULT_DB_PATH", str(tmp_path / "vault.db"))
    monkeypatch.delenv("VAULT_ADMIN_EMAIL", raising=False)
    monkeypatch.delenv("VAULT_ADMIN_PASSWORD", raising=False)
    monkeypatch.delenv("VAULT_ADMIN_NAME", raising=False)
    return tmp_path / "vault.db"


class TestBootstrapAdmin:
    def test_creates_the_first_administrator(self, _isolated_db, monkeypatch, capsys):
        monkeypatch.setenv("VAULT_ADMIN_EMAIL", "Ana@Acme.com")
        monkeypatch.setenv("VAULT_ADMIN_PASSWORD", PASSWORD)

        assert vault_main.bootstrap_admin([]) == 0

        store = vs.VaultStore(_isolated_db)
        admin = store.get_user_by_email("ana@acme.com")
        assert admin["is_admin"] == 1
        assert sec.verify_password(admin["password_hash"], PASSWORD)
        assert "administrator created" in capsys.readouterr().out

    def test_password_is_removed_from_the_environment(self, monkeypatch):
        monkeypatch.setenv("VAULT_ADMIN_EMAIL", "ana@acme.com")
        monkeypatch.setenv("VAULT_ADMIN_PASSWORD", PASSWORD)
        vault_main.bootstrap_admin([])
        import os

        assert "VAULT_ADMIN_PASSWORD" not in os.environ

    def test_is_idempotent_and_updates_the_password(self, _isolated_db, monkeypatch):
        monkeypatch.setenv("VAULT_ADMIN_EMAIL", "ana@acme.com")
        monkeypatch.setenv("VAULT_ADMIN_PASSWORD", PASSWORD)
        vault_main.bootstrap_admin([])

        monkeypatch.setenv("VAULT_ADMIN_PASSWORD", "outra senha bem longa")
        assert vault_main.bootstrap_admin([]) == 0

        store = vs.VaultStore(_isolated_db)
        admin = store.get_user_by_email("ana@acme.com")
        assert sec.verify_password(admin["password_hash"], "outra senha bem longa")
        assert len(store.list_users()) == 1

    def test_refuses_to_promote_an_existing_non_admin(self, _isolated_db, monkeypatch):
        store = vs.VaultStore(_isolated_db)
        store.create_user(email="carlos@cliente.dev")

        monkeypatch.setenv("VAULT_ADMIN_EMAIL", "carlos@cliente.dev")
        monkeypatch.setenv("VAULT_ADMIN_PASSWORD", PASSWORD)
        assert vault_main.bootstrap_admin([]) == 2
        assert store.get_user_by_email("carlos@cliente.dev")["is_admin"] == 0

    def test_invalid_email_is_refused(self, monkeypatch):
        monkeypatch.setenv("VAULT_ADMIN_EMAIL", "not-an-email")
        monkeypatch.setenv("VAULT_ADMIN_PASSWORD", PASSWORD)
        assert vault_main.bootstrap_admin([]) == 2

    def test_short_password_is_refused(self, monkeypatch):
        monkeypatch.setenv("VAULT_ADMIN_EMAIL", "ana@acme.com")
        monkeypatch.setenv("VAULT_ADMIN_PASSWORD", "short")
        assert vault_main.bootstrap_admin([]) == 2

    def test_email_can_come_from_the_command_line(self, _isolated_db, monkeypatch):
        monkeypatch.setenv("VAULT_ADMIN_PASSWORD", PASSWORD)
        assert vault_main.bootstrap_admin(["ana@acme.com"]) == 0
        assert vs.VaultStore(_isolated_db).count_admins() == 1


class TestCli:
    def test_unknown_command_exits_nonzero(self, monkeypatch, capsys):
        monkeypatch.setattr("sys.argv", ["deepmem0-vault", "frobnicate"])
        assert vault_main.main() == 2
        assert "unknown command" in capsys.readouterr().err

    def test_help(self, monkeypatch, capsys):
        monkeypatch.setattr("sys.argv", ["deepmem0-vault", "--help"])
        assert vault_main.main() == 0
        assert "usage" in capsys.readouterr().out

    def test_bare_invocation_serves(self, monkeypatch):
        called = {}
        monkeypatch.setattr("sys.argv", ["deepmem0-vault"])
        monkeypatch.setattr(vault_main, "serve", lambda: called.setdefault("served", True) or 0)
        vault_main.main()
        assert called["served"]


class TestPackagedAssets:
    """Templates and static files must ship with the package, not just the repo."""

    def test_templates_and_static_are_inside_the_package(self):
        from mem0_mcp_selfhosted.vault import web

        assert (web.TEMPLATES_DIR / "login.html").is_file()
        assert (web.TEMPLATES_DIR / "shell.html").is_file()
        assert (web.TEMPLATES_DIR / "partials" / "token_modal.html").is_file()
        assert (web.STATIC_DIR / "vault.css").is_file()
        assert (web.STATIC_DIR / "htmx.min.js").is_file(), "HTMX must be vendored, not a CDN"

    def test_every_template_renders_without_undefined_syntax(self):
        from jinja2 import Environment, FileSystemLoader

        from mem0_mcp_selfhosted.vault import web

        env = Environment(loader=FileSystemLoader(str(web.TEMPLATES_DIR)), autoescape=True)
        for template in web.TEMPLATES_DIR.rglob("*.html"):
            env.get_template(str(template.relative_to(web.TEMPLATES_DIR)))

    def test_autoescape_is_on(self):
        from mem0_mcp_selfhosted.vault import store as store_mod
        from mem0_mcp_selfhosted.vault import web

        routes = web.Routes(store_mod.VaultStore(":memory:"))
        assert routes.templates.env.autoescape


class TestBootstrapDoesNotMutateOnRefusal:
    """A refused bootstrap must leave the account exactly as it was.

    Found by independent review: the original order set the password first and
    only then discovered the account was not an administrator.
    """

    def test_refusing_a_non_admin_leaves_the_password_untouched(self, _isolated_db, monkeypatch):
        store = vs.VaultStore(_isolated_db)
        original = sec.hash_password("a senha original do carlos")
        store.create_user(email="carlos@cliente.dev", password_hash=original)

        monkeypatch.setenv("VAULT_ADMIN_EMAIL", "carlos@cliente.dev")
        monkeypatch.setenv("VAULT_ADMIN_PASSWORD", PASSWORD)
        assert vault_main.bootstrap_admin([]) == 2

        after = store.get_user_by_email("carlos@cliente.dev")["password_hash"]
        assert after == original
        assert sec.verify_password(after, "a senha original do carlos")
        assert not sec.verify_password(after, PASSWORD)

    def test_bootstrap_recovers_a_disabled_admin(self, _isolated_db, monkeypatch):
        """Otherwise the last admin disabling themselves is a permanent lockout."""
        store = vs.VaultStore(_isolated_db)
        uid = store.create_user(
            email="ana@acme.com", is_admin=True, password_hash=sec.hash_password(PASSWORD),
        )
        store.set_user_disabled(uid, True)

        monkeypatch.setenv("VAULT_ADMIN_EMAIL", "ana@acme.com")
        monkeypatch.setenv("VAULT_ADMIN_PASSWORD", "senha nova de recuperacao")
        assert vault_main.bootstrap_admin([]) == 0

        recovered = store.get_user_by_email("ana@acme.com")
        assert recovered["disabled_at"] is None
        assert sec.verify_password(recovered["password_hash"], "senha nova de recuperacao")


class TestPromotionCheck:
    """The shadow->on gate as a command, so a script can depend on it."""

    def _vault_with(self, path, *, used, silent, denials=0):
        from datetime import datetime, timedelta, timezone

        store = vs.VaultStore(path)
        uid = store.create_user(email="client@example.com")
        recent = (datetime.now(timezone.utc) - timedelta(minutes=10)).isoformat()
        for _ in range(used):
            tid = store.create_token(user_id=uid, token=sec.generate_token())["id"]
            with store._tx() as conn:
                conn.execute("UPDATE tokens SET last_used_at = ? WHERE id = ?", (recent, tid))
        for _ in range(silent):
            store.create_token(user_id=uid, token=sec.generate_token())
        if denials:
            store.record_denials({"unknown": denials})
        return store

    def test_ready_when_every_token_was_seen_and_nothing_was_denied(
        self, _isolated_db, capsys
    ):
        self._vault_with(_isolated_db, used=2, silent=0)
        assert vault_main.promotion_check([]) == 0
        assert "READY" in capsys.readouterr().out

    def test_not_ready_while_a_token_is_silent(self, _isolated_db, capsys):
        self._vault_with(_isolated_db, used=1, silent=1)
        assert vault_main.promotion_check([]) == 1
        out = capsys.readouterr()
        assert "WOULD 401" in out.out
        assert "NOT READY" in out.err

    def test_not_ready_while_denials_exist(self, _isolated_db, capsys):
        self._vault_with(_isolated_db, used=1, silent=0, denials=3)
        assert vault_main.promotion_check([]) == 1
        assert "unknown: 3" in capsys.readouterr().out

    def test_an_empty_vault_is_never_ready(self, _isolated_db):
        vs.VaultStore(_isolated_db)
        assert vault_main.promotion_check([]) == 1

    def test_window_is_configurable(self, _isolated_db, capsys):
        self._vault_with(_isolated_db, used=1, silent=0)
        assert vault_main.promotion_check(["--window-hours", "1"]) == 0
        assert "last 1h" in capsys.readouterr().out

    def test_bad_window_is_refused(self, _isolated_db):
        vs.VaultStore(_isolated_db)
        assert vault_main.promotion_check(["--window-hours", "abc"]) == 2


class TestDenialWindow:
    """Negação de ontem é histórico; negação de agora é problema.

    A primeira versão contava negações na janela inteira de 72h e ficava
    NOT READY por três dias por causa dos 401 da própria migração que ela
    tinha orientado — reprovava a transição em vez de medir o estado atual.
    """

    def _vault(self, path, *, denial_age_hours):
        from datetime import datetime, timedelta, timezone

        store = vs.VaultStore(path)
        uid = store.create_user(email="client@example.com")
        tid = store.create_token(user_id=uid, token=sec.generate_token())["id"]
        recent = (datetime.now(timezone.utc) - timedelta(minutes=5)).isoformat()
        with store._tx() as conn:
            conn.execute("UPDATE tokens SET last_used_at = ? WHERE id = ?", (recent, tid))
            moment = datetime.now(timezone.utc) - timedelta(hours=denial_age_hours)
            conn.execute(
                "INSERT INTO auth_denials (bucket_start, reason, count, last_seen_at)"
                " VALUES (?, 'missing', 40, ?)",
                (moment.isoformat(), moment.isoformat()),
            )
        return store

    def test_old_denials_do_not_veto(self, _isolated_db, capsys):
        self._vault(_isolated_db, denial_age_hours=20)
        assert vault_main.promotion_check([]) == 0
        out = capsys.readouterr().out
        assert "READY" in out
        assert "contexto, não veto" in out, "o histórico deve aparecer, sem reprovar"

    def test_recent_denials_still_veto(self, _isolated_db, capsys):
        self._vault(_isolated_db, denial_age_hours=0)
        assert vault_main.promotion_check([]) == 1
        assert "missing: 40" in capsys.readouterr().out


class TestOnDemandTokens:
    """Cliente periódico não pode travar o portão para sempre.

    O harness de eval roda quando alguém o roda. Sem uma forma de dizer isso,
    ou ele fica sem token (e o gate de qualidade morre) ou o promotion-check
    fica NOT READY eternamente. A exceção precisa ter NOME, não ser silêncio.
    """

    def _issue(self, argv, capsys):
        code = vault_main.issue_token(argv)
        return code, capsys.readouterr()

    def test_on_demand_token_does_not_veto(self, _isolated_db, capsys):
        from datetime import datetime, timedelta, timezone

        store = vs.VaultStore(_isolated_db)
        uid = store.create_user(email="client@example.com")
        tid = store.create_token(user_id=uid, token=sec.generate_token())["id"]
        recent = (datetime.now(timezone.utc) - timedelta(minutes=5)).isoformat()
        with store._tx() as conn:
            conn.execute("UPDATE tokens SET last_used_at = ? WHERE id = ?", (recent, tid))

        # nunca usado, mas marcado como por demanda
        store.create_token(
            user_id=uid, token=sec.generate_token(), label="eval-harness", on_demand=True
        )

        assert vault_main.promotion_check([]) == 0
        out = capsys.readouterr().out
        assert "por demanda: 1" in out, "tem de aparecer na listagem"
        assert "READY" in out

    def test_a_normal_token_still_vetoes(self, _isolated_db, capsys):
        store = vs.VaultStore(_isolated_db)
        uid = store.create_user(email="client@example.com")
        store.create_token(user_id=uid, token=sec.generate_token(), label="esquecido")
        assert vault_main.promotion_check([]) == 1

    def test_cli_flag_marks_the_token(self, _isolated_db, capsys):
        code, out = self._issue(
            ["--email", "eval@t430.test", "--label", "eval-harness", "--on-demand"], capsys
        )
        assert code == 0
        assert "por demanda" in out.out
        store = vs.VaultStore(_isolated_db)
        token = store.list_tokens(store.get_user_by_email("eval@t430.test")["id"])[0]
        assert token["on_demand"] == 1

    def test_without_the_flag_it_is_a_normal_client(self, _isolated_db, capsys):
        self._issue(["--email", "eval@t430.test", "--label", "normal"], capsys)
        store = vs.VaultStore(_isolated_db)
        token = store.list_tokens(store.get_user_by_email("eval@t430.test")["id"])[0]
        assert token["on_demand"] == 0
