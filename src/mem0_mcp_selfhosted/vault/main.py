"""Entry point for the vault service (:8080) and its bootstrap command.

    deepmem0-vault                  # serve the UI
    deepmem0-vault bootstrap-admin  # create/refresh the first administrator

Bootstrap is an explicit command, never a boot-time side effect: a service
that reconciles an admin password from the environment on every restart turns
a stale env file into a silent credential reset.
"""

from __future__ import annotations

import getpass
import logging
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

from mem0_mcp_selfhosted.env import env
from mem0_mcp_selfhosted.vault import security
from mem0_mcp_selfhosted.vault import store as vs

logger = logging.getLogger(__name__)

USAGE = (
    "usage: deepmem0-vault [serve]\n"
    "       deepmem0-vault bootstrap-admin [email]\n"
    "       deepmem0-vault issue-token --email E --label L [--name N]\n"
    "                                  [--expires-days D] [--bind-scope S]\n"
    "       deepmem0-vault promotion-check [--window-hours N]"
)


def _db_path() -> str:
    return env("MEM0_VAULT_DB_PATH") or str(Path.home() / ".mem0" / "vault.db")


def _load_env() -> None:
    """Load .env / .vault.env when running by hand (systemd uses EnvironmentFile)."""
    try:
        from dotenv import load_dotenv
    except ImportError:  # pragma: no cover - dotenv ships with the package
        return
    load_dotenv()
    vault_env = Path.cwd() / ".vault.env"
    if vault_env.exists():
        load_dotenv(vault_env)


def bootstrap_admin(argv: list[str]) -> int:
    """Create (or update the password of) the first administrator. Idempotent."""
    store = vs.VaultStore(_db_path())

    email_raw = env("VAULT_ADMIN_EMAIL") or (argv[0] if argv else "")
    if not email_raw:
        email_raw = input("admin email: ").strip()
    try:
        email = security.normalize_email(email_raw)
    except security.ValidationError as exc:
        print(f"error: invalid email ({exc})", file=sys.stderr)
        return 2

    password = os.environ.pop("VAULT_ADMIN_PASSWORD", "").strip()
    if not password:
        password = getpass.getpass("admin password: ")
        if password != getpass.getpass("repeat password: "):
            print("error: passwords do not match", file=sys.stderr)
            return 2
    try:
        password_hash = security.hash_password(password)
    except security.ValidationError:
        print("error: password must be at least 8 characters", file=sys.stderr)
        return 2
    finally:
        del password

    existing = store.get_user_by_email(email)
    if existing:
        # Check BEFORE mutating: refusing to promote must not have already
        # changed the account's password on the way out.
        if not existing["is_admin"]:
            print(
                f"error: {email} exists but is not an administrator", file=sys.stderr
            )
            return 2
        store.set_password_hash(existing["id"], password_hash)
        if existing["disabled_at"]:
            # Otherwise the last admin can lock everyone out permanently.
            store.set_user_disabled(existing["id"], False, actor_email="bootstrap")
            print(f"administrator re-enabled: {email}")
        print(f"password updated for {email}")
        return 0

    display_name = env("VAULT_ADMIN_NAME") or email.split("@")[0]
    store.create_user(
        email=email, display_name=display_name, is_admin=True,
        password_hash=password_hash, actor_email="bootstrap",
    )
    print(f"administrator created: {email}")
    return 0


def issue_token(argv: list[str]) -> int:
    """Mint a token from the server's own terminal.

    The UI reveals a token over the LAN, which today is plain HTTP — the most
    valuable secret in the system crossing the wire in the clear. Issued here,
    the plaintext never leaves the box. The UI keeps the rest (listing,
    rotation, revocation, audit), which is all prefix-only.

        deepmem0-vault issue-token --email dev@lan --label claude-code-local
                                   [--name "Dev"] [--expires-days 90]
                                   [--bind-scope alice]
    """
    flags = {"--on-demand"}
    on_demand = "--on-demand" in argv
    argv = [a for a in argv if a not in flags]
    options = _parse_options(argv, {
        "--email", "--label", "--name", "--expires-days", "--bind-scope",
    })
    if options is None:
        return 2

    email_raw = options.get("--email", "")
    if not email_raw:
        print("error: --email is required", file=sys.stderr)
        return 2
    try:
        email = security.normalize_email(email_raw)
        label = security.normalize_label(options.get("--label", ""))
    except security.ValidationError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    expires_at = None
    if "--expires-days" in options:
        try:
            days = int(options["--expires-days"])
            if days <= 0:
                raise ValueError
        except ValueError:
            print("error: --expires-days needs a positive number", file=sys.stderr)
            return 2
        expires_at = (
            datetime.now(timezone.utc) + timedelta(days=days)
        ).isoformat()

    store = vs.VaultStore(_db_path())
    user = store.get_user_by_email(email)
    if user is None:
        user_id = store.create_user(
            email=email, display_name=options.get("--name") or email.split("@")[0],
            mem0_user_id=options.get("--bind-scope", ""), actor_email="cli",
        )
        print(f"created user {email}")
    else:
        if user["disabled_at"]:
            print(f"error: {email} is disabled — re-enable it first", file=sys.stderr)
            return 2
        if options.get("--bind-scope") and user["mem0_user_id"] != options["--bind-scope"]:
            print(
                f"error: {email} is bound to '{user['mem0_user_id']}', refusing to "
                f"rebind to '{options['--bind-scope']}' (create a separate user)",
                file=sys.stderr,
            )
            return 2
        user_id = user["id"]

    for _attempt in range(3):
        token = security.generate_token()
        try:
            created = store.create_token(
                user_id=user_id, token=token, label=label, expires_at=expires_at,
                on_demand=on_demand, actor_email="cli",
            )
            break
        except vs.TokenCollision:
            continue
    else:
        print("error: could not mint a unique token", file=sys.stderr)
        return 1

    print()
    print(token)
    print()
    print(f"  user:    {email}")
    print(f"  label:   {label}")
    print(f"  prefix:  {created['prefix']}… (this is all the UI will show from now on)")
    print(f"  expires: {expires_at or 'never'}")
    if on_demand:
        print("  uso:     por demanda (não veta o promotion-check quando ocioso)")
    scope = store.get_user(user_id)["mem0_user_id"]
    print(f"  scope:   {scope or 'unbound (authentication only)'}")
    print()
    print("Shown once. Configure the client now, e.g.:")
    print('  claude mcp add --scope user --transport http deepmem0 '
          'http://localhost:8081/mcp \\')
    print(f'    --header "Authorization: Bearer {token}"')
    return 0


def _parse_options(argv: list[str], allowed: set[str]) -> dict[str, str] | None:
    """Tiny --key value parser; unknown flags are an error, not a surprise."""
    options: dict[str, str] = {}
    index = 0
    while index < len(argv):
        key = argv[index]
        if key not in allowed:
            print(f"error: unknown option {key}", file=sys.stderr)
            return None
        if index + 1 >= len(argv):
            print(f"error: {key} needs a value", file=sys.stderr)
            return None
        options[key] = argv[index + 1]
        index += 2
    return options


def promotion_check(argv: list[str]) -> int:
    """Is it safe to flip MEM0_REQUIRE_AUTH from shadow to on?

    Exit 0 = ready, 1 = not ready — so it can gate a deployment script instead
    of relying on somebody reading a dashboard. "Not ready" is the default:
    silence is not evidence.
    """
    window = 72
    if "--window-hours" in argv:
        try:
            window = max(1, int(argv[argv.index("--window-hours") + 1]))
        except (IndexError, ValueError):
            print("error: --window-hours needs a number", file=sys.stderr)
            return 2

    store = vs.VaultStore(_db_path(), create=False)
    report = store.promotion_readiness(window_hours=window)

    print(f"window: last {report['window_hours']}h")
    print(f"active tokens: {report['expected']}")
    print(f"  seen:   {len(report['seen'])}")
    for token in report["seen"]:
        print(f"    ok    {token['prefix']}… {token['email']} ({token['use_count']} uses)")
    if report["on_demand"]:
        print(f"  por demanda: {len(report['on_demand'])} (listados, não vetam)")
        for token in report["on_demand"]:
            print(f"    ~     {token['prefix']}… {token['email']} ({token['label']})")
    print(f"  silent: {len(report['silent'])}")
    for token in report["silent"]:
        print(f"    WOULD 401  {token['prefix']}… {token['email']} — never used in window")
    print(f"denials (last {report['denial_window_hours']}h): {report['denial_total']}")
    for denial in report["denials"]:
        # O CLIENTE importa tanto quanto o motivo: sem ele não dá para separar
        # "meu próprio teste no loopback" de "cliente de verdade quebrado".
        origem = denial.get("last_client") or "?"
        print(f"    {denial['reason']}: {denial['count']} de {origem} "
              f"(última {denial['last_seen_at'][11:19]})")
    if report["denial_historic_total"] > report["denial_total"]:
        print(f"  (histórico na janela de {report['window_hours']}h: "
              f"{report['denial_historic_total']} — contexto, não veto)")

    if report["ready"]:
        print("\nREADY — every active token authorized recently, zero denials.")
        return 0
    print("\nNOT READY — fix the above before MEM0_REQUIRE_AUTH=on.", file=sys.stderr)
    return 1


def serve() -> int:
    import uvicorn

    from mem0_mcp_selfhosted.vault.web import create_app

    host = env("VAULT_HOST", "0.0.0.0")
    port = int(env("VAULT_PORT", "8080"))
    app = create_app(_db_path())

    store: vs.VaultStore = app.state.store
    if store.count_admins() == 0:
        # Not fatal: the login page explains how to bootstrap. A vault that
        # refuses to start is harder to fix than one that says what's missing.
        logger.warning(
            "no administrator in %s — run 'deepmem0-vault bootstrap-admin'", _db_path()
        )

    logger.info("DeepMem0 Vault on http://%s:%s (db=%s)", host, port, _db_path())
    uvicorn.run(
        app, host=host, port=port, workers=1,
        log_level=env("VAULT_LOG_LEVEL", "info").lower(),
        # See the note in server.py: the audit log records request.client.host,
        # which behind a proxy is only the real client if uvicorn is trusted to
        # rewrite it — and only from the proxy's address.
        proxy_headers=True,
        forwarded_allow_ips=env("VAULT_FORWARDED_ALLOW_IPS", "127.0.0.1"),
    )
    return 0


def main() -> int:
    logging.basicConfig(
        level=getattr(logging, env("VAULT_LOG_LEVEL", "INFO").upper(), logging.INFO),
        format="%(levelname)s %(name)s | %(message)s",
    )
    _load_env()

    argv = sys.argv[1:]
    if not argv:
        return serve()
    command, rest = argv[0], argv[1:]
    if command == "bootstrap-admin":
        return bootstrap_admin(rest)
    if command == "promotion-check":
        return promotion_check(rest)
    if command == "issue-token":
        return issue_token(rest)
    if command in ("-h", "--help", "help"):
        print(USAGE)
        return 0
    print(f"unknown command: {command}\n{USAGE}", file=sys.stderr)
    return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
