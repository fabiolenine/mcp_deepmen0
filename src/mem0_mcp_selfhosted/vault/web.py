"""The vault UI (:8080): admin login, users, tokens, audit log.

Server-rendered Jinja2 + HTMX. The only client-side behavior that matters is
the one-time token reveal, which is why token creation/rotation returns an
HTMX partial instead of redirecting: the plaintext exists in exactly one
response, is never in a URL, never in a log, and is never re-rendered.

Security posture (LAN service, no TLS — see the plan):
- session cookie signed with VAULT_SECRET_KEY, 12 h, HttpOnly + SameSite=Lax,
  rotated on login;
- constant ~2 s delay on failed login instead of a lockout (a lockout on a
  single-admin service is a denial-of-service handed to the attacker);
- CSRF token per session on every POST;
- ``Cache-Control: no-store`` + anti-sniff/framing headers on every
  authenticated response.
"""

from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone
from functools import wraps
from pathlib import Path
from typing import Any, Callable

import anyio
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.sessions import SessionMiddleware
from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse, RedirectResponse, Response
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles
from starlette.templating import Jinja2Templates

from mem0_mcp_selfhosted.env import env
from mem0_mcp_selfhosted.vault import i18n, security
from mem0_mcp_selfhosted.vault import store as vs
from mem0_mcp_selfhosted.vault.middleware import normalize_mode

logger = logging.getLogger(__name__)

HERE = Path(__file__).parent
TEMPLATES_DIR = HERE / "templates"
STATIC_DIR = HERE / "static"

SESSION_MAX_AGE = 12 * 3600
FAILED_LOGIN_DELAY_S = 2.0
MAX_FORM_BYTES = 16 * 1024
AUDIT_PAGE_SIZE = 40

SECURE_HEADERS = {
    "Cache-Control": "no-store",
    "X-Frame-Options": "DENY",
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "same-origin",
    # Tudo servido pela própria origem. A UI não busca NADA de fora — nem
    # fonte, nem script — então a política pode ser fechada sem exceção.
    # 'unsafe-inline' em style- porque os templates ainda têm style="" pontual
    # herdado do mockup; script-src NÃO precisa dele (o clipboard virou arquivo).
    "Content-Security-Policy": (
        "default-src 'none'; "
        "script-src 'self'; "
        "style-src 'self' 'unsafe-inline'; "
        "font-src 'self'; "
        "img-src 'self' data:; "
        "connect-src 'self'; "
        "form-action 'self'; "
        "base-uri 'none'; "
        "frame-ancestors 'none'"
    ),
}


# ---------------------------------------------------------------- helpers


def _grace_hours() -> int:
    try:
        return max(0, int(env("VAULT_TOKEN_GRACE_HOURS", "24")))
    except ValueError:
        return 24


def _https_only() -> bool:
    """Is the vault served over TLS? Drives the session cookie's Secure flag.

    Strict on purpose: a typo here must NOT resolve to False. The generic
    bool_env() helper treats anything unrecognized as false, which is the
    unsafe direction for a flag whose job is to keep the session cookie off
    plaintext connections — same lesson as MEM0_REQUIRE_AUTH.
    """
    raw = env("VAULT_HTTPS_ONLY").lower()
    if raw in ("", "false", "0", "no"):
        return False
    if raw in ("true", "1", "yes"):
        return True
    raise ValueError(f"VAULT_HTTPS_ONLY={raw!r} is not a boolean (true/false)")


def _session_cookie_name(https_only: bool) -> str:
    """Cookie name changes with the scheme, so the cutover invalidates sessions.

    Flipping https_only alone would leave every already-issued cookie — signed,
    valid, and WITHOUT the Secure flag — usable until it expired. Renaming
    forces a fresh login and orphans the insecure ones.
    """
    return "vault_session_s" if https_only else "vault_session"


def _readiness_window_hours() -> int:
    """How far back the shadow->on evidence is read (default 3 days)."""
    try:
        return max(1, int(env("VAULT_READINESS_WINDOW_HOURS", "72")))
    except ValueError:
        return 72


def _denial_window_hours() -> float:
    """Negações recentes o bastante para vetar (o resto é histórico)."""
    try:
        return max(0.1, float(env("VAULT_DENIAL_WINDOW_HOURS", "1")))
    except ValueError:
        return 1.0


def _mcp_health_url() -> str:
    return env("VAULT_MCP_HEALTH_URL", "http://127.0.0.1:8081/health")


def _auth_mode_label() -> tuple[str, bool]:
    """``(modo, confirmado_pelo_mcp)``.

    Preferimos o que o :8081 reporta. Se ele estiver fora do ar, caímos no env
    local — mas o segundo valor vira False, e a UI diz que não está confirmado
    em vez de afirmar uma postura de segurança que ninguém verificou.
    """
    reported = (_mcp_health().get("detail") or {}).get("auth_mode")
    if reported in ("off", "shadow", "on"):
        return reported, True
    try:
        return normalize_mode(env("MEM0_REQUIRE_AUTH")), False
    except ValueError:
        return "invalid", False


#: Cache curto do /health: o modo de auth é lido em TODA página (fica na
#: sidebar), e uma chamada HTTP por render seria caro e frágil.
_HEALTH_TTL_S = 20.0
_health_cache: dict[str, Any] = {"at": 0.0, "value": None}


def _mcp_health(*, cached: bool = True) -> dict[str, Any]:
    """Best-effort peek at the MCP the vault protects (dashboard tile).

    Também é a FONTE DA VERDADE do modo de auth: quem sabe se o gate está
    exigindo token é o processo que o aplica, não uma cópia da variável de
    ambiente no serviço vizinho. Duplicar o env faria a UI mentir no dia em
    que alguém trocasse um lado e esquecesse o outro — que foi exatamente o
    que aconteceu em 20/07/2026, com o cofre exibindo "off" enquanto o MCP
    já respondia 401.
    """
    now = time.monotonic()
    if cached and _health_cache["value"] is not None and now - _health_cache["at"] < _HEALTH_TTL_S:
        return _health_cache["value"]
    result = _probe_mcp_health()
    _health_cache.update(at=now, value=result)
    return result


def _probe_mcp_health() -> dict[str, Any]:
    try:
        with urllib.request.urlopen(_mcp_health_url(), timeout=1.5) as resp:
            payload = json.loads(resp.read().decode("utf-8") or "{}")
            return {"ok": resp.status == 200, "code": resp.status, "detail": payload}
    except urllib.error.HTTPError as exc:
        return {"ok": False, "code": exc.code, "detail": {}}
    except Exception:  # noqa: BLE001 - a down MCP is information, not an error
        return {"ok": False, "code": None, "detail": {}}


def _parse(ts: str | None) -> datetime | None:
    if not ts:
        return None
    try:
        parsed = datetime.fromisoformat(ts)
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def humanize(ts: str | None, lang: str) -> str:
    """Relative time as in the design ("há 4 min" / "4 min ago")."""
    moment = _parse(ts)
    if moment is None:
        return "—"
    delta = (datetime.now(timezone.utc) - moment).total_seconds()
    future = delta < 0
    delta = abs(delta)
    if delta < 60:
        value, unit = int(delta), "s"
    elif delta < 3600:
        value, unit = int(delta // 60), "min"
    elif delta < 86400:
        value, unit = int(delta // 3600), "h"
    else:
        value, unit = int(delta // 86400), "d"

    if lang == "en":
        plural = "s" if unit == "d" and value != 1 else ""
        unit_en = {"s": "s", "min": "min", "h": "h", "d": "day"}[unit] + plural
        return f"in {value} {unit_en}" if future else f"{value} {unit_en} ago"
    unit_pt = {"s": "s", "min": "min", "h": "h", "d": "dia" + ("s" if value != 1 else "")}[unit]
    return f"em {value} {unit_pt}" if future else f"há {value} {unit_pt}"


def token_status(token: dict[str, Any]) -> str:
    """active | expiring | revoked — the three badges in the design."""
    if token.get("revoked_at"):
        return "revoked"
    expires = _parse(token.get("expires_at"))
    if expires is None:
        return "active"
    if expires <= datetime.now(timezone.utc):
        return "revoked"  # expired reads as dead in the UI
    if expires - datetime.now(timezone.utc) <= timedelta(hours=48):
        return "expiring"
    return "active"


def audit_kind(action: str) -> str:
    if action.endswith(".create"):
        return "ok"
    if action.endswith(".rotate"):
        return "warn"
    if action.endswith(".revoke") or action.endswith(".disable"):
        return "danger"
    return "info"


# ---------------------------------------------------------------- app state


class VaultWeb:
    """Holds the store + templates; routes are thin methods over it."""

    def __init__(self, store: vs.VaultStore):
        self.store = store
        self.templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
        self.templates.env.globals.update(
            initials=security.initials,
            avatar_hue=security.avatar_hue,
            humanize=humanize,
            token_status=token_status,
            audit_kind=audit_kind,
        )

    # -- request context ------------------------------------------------

    def lang(self, request: Request) -> str:
        return i18n.normalize_lang(
            request.query_params.get("lang") or request.cookies.get(i18n.LANG_COOKIE)
        )

    def context(self, request: Request, **extra: Any) -> dict[str, Any]:
        lang = self.lang(request)
        mode, mode_confirmed = _auth_mode_label()
        ctx = {
            "request": request,
            "t": i18n.strings(lang),
            "lang": lang,
            "auth_mode": mode,
            "auth_mode_confirmed": mode_confirmed,
            "mode_hint_key": {
                "off": "modeHintOff", "shadow": "modeHintShadow", "on": "modeHintOn",
            }.get(mode, "modeHintShadow"),
            "csrf": request.session.get("csrf", ""),
            "admin": {
                "id": request.session.get("uid"),
                "email": request.session.get("email", ""),
                "name": request.session.get("name", ""),
            },
            "grace_hours": _grace_hours(),
            "nav": extra.pop("nav", "dash"),
        }
        ctx.update(extra)
        return ctx

    def render(
        self, request: Request, template: str, status_code: int = 200, **extra: Any
    ) -> Response:
        response = self.templates.TemplateResponse(
            request, template, self.context(request, **extra), status_code=status_code
        )
        self._harden(request, response)
        return response

    def _harden(self, request: Request, response: Response) -> None:
        for header, value in SECURE_HEADERS.items():
            response.headers[header] = value
        chosen = request.query_params.get("lang")
        if chosen and i18n.normalize_lang(chosen) == chosen:
            response.set_cookie(
                i18n.LANG_COOKIE, chosen, max_age=365 * 24 * 3600,
                httponly=False, samesite="lax",
            )

    def redirect(self, request: Request, url: str) -> Response:
        response = RedirectResponse(url, status_code=303)
        self._harden(request, response)
        return response

    # -- auth plumbing --------------------------------------------------

    def current_admin(self, request: Request) -> dict[str, Any] | None:
        uid = request.session.get("uid")
        if not uid:
            return None
        user = self.store.get_user(int(uid))
        if not user or user["disabled_at"] or not user["is_admin"]:
            request.session.clear()
            return None
        # O cookie é assinado no cliente: sem esta comparação, logout não
        # invalida nada e uma cópia roubada vale até expirar sozinha.
        if request.session.get("epoch") != user["session_epoch"]:
            request.session.clear()
            return None
        return user

    async def form(self, request: Request) -> dict[str, str]:
        body = await request.body()
        if len(body) > MAX_FORM_BYTES:
            raise ValueError("form too large")
        return {k: str(v) for k, v in (await request.form()).items()}

    def csrf_guard(self, request: Request, form: dict[str, str]) -> bool:
        return security.csrf_ok(request.session.get("csrf"), form.get("csrf"))

    def client_ip(self, request: Request) -> str:
        return request.client.host if request.client else ""


def login_required(handler: Callable) -> Callable:
    @wraps(handler)
    async def wrapper(self: VaultWeb, request: Request) -> Response:
        if self.current_admin(request) is None:
            return self.redirect(request, "/login")
        return await handler(self, request)

    return wrapper


# ---------------------------------------------------------------- routes


class Routes(VaultWeb):
    # -- session --------------------------------------------------------

    async def login_page(self, request: Request) -> Response:
        if self.current_admin(request) is not None:
            return self.redirect(request, "/")
        if not request.session.get("csrf"):
            request.session["csrf"] = security.new_csrf_token()
        return self.render(
            request, "login.html",
            has_admin=self.store.count_admins() > 0,
            error=request.query_params.get("error", ""),
        )

    async def login_submit(self, request: Request) -> Response:
        form = await self.form(request)
        lang = self.lang(request)
        if not self.csrf_guard(request, form):
            return self.render(
                request, "login.html", status_code=403,
                has_admin=self.store.count_admins() > 0,
                error=i18n.error_message(lang, "csrf"),
            )

        email = (form.get("email") or "").strip().lower()
        password = form.get("password") or ""
        user = self.store.get_user_by_email(email) if email else None
        ok = (
            user is not None
            and user["is_admin"]
            and not user["disabled_at"]
            and security.verify_password(user["password_hash"], password)
        )

        if not ok:
            # Constant delay: no lockout to weaponize, no timing oracle.
            # anyio.sleep, not time.sleep — this handler is async, and blocking
            # the loop for 2 s per attempt hands an attacker a trivial DoS.
            await anyio.sleep(FAILED_LOGIN_DELAY_S)
            self.store.record_event(
                action="login.failed", actor_email=email, ip=self.client_ip(request),
                success=False, subject_type="user",
                subject_id=str(user["id"]) if user else "",
            )
            return self.render(
                request, "login.html", status_code=401,
                has_admin=self.store.count_admins() > 0,
                error=i18n.error_message(lang, "bad_credentials"),
            )

        request.session.clear()  # rotate the session on privilege change
        request.session.update({
            "uid": user["id"],
            "email": user["email"],
            "name": user["display_name"] or user["email"],
            "epoch": user["session_epoch"],
            "csrf": security.new_csrf_token(),
        })
        self.store.record_event(
            action="login.ok", actor_id=user["id"], actor_email=user["email"],
            ip=self.client_ip(request), subject_type="user", subject_id=str(user["id"]),
        )
        return self.redirect(request, "/")

    async def logout(self, request: Request) -> Response:
        form = await self.form(request)
        if self.csrf_guard(request, form):
            uid = request.session.get("uid")
            request.session.clear()
            if uid:
                # Não basta limpar a sessão local: incrementa a época para que
                # QUALQUER cópia daquele cookie pare de valer agora.
                self.store.bump_session_epoch(int(uid), reason="logout")
        return self.redirect(request, "/login")

    # -- dashboard ------------------------------------------------------

    @login_required
    async def dashboard(self, request: Request) -> Response:
        users = self.store.list_users()
        stats = self.store.stats()
        cutoff = (datetime.now(timezone.utc) - timedelta(hours=24)).isoformat()
        seen_24h = self.store.count_tokens_used_since(cutoff)
        return self.render(
            request, "dashboard.html", nav="dash",
            users=users, stats=stats, seen_24h=seen_24h, mcp=_mcp_health(),
            readiness=self.store.promotion_readiness(
                window_hours=_readiness_window_hours(),
                denial_window_hours=_denial_window_hours(),
            ),
            new_user_open=bool(request.query_params.get("new")),
            error=request.query_params.get("error", ""),
        )

    @login_required
    async def create_user(self, request: Request) -> Response:
        admin = self.current_admin(request)
        form = await self.form(request)
        lang = self.lang(request)
        if not self.csrf_guard(request, form):
            return self.redirect(request, f"/?new=1&error={i18n.error_message(lang, 'csrf')}")
        try:
            name = security.normalize_name(form.get("display_name", ""))
            email = security.normalize_email(form.get("email", ""))
            self.store.create_user(
                email=email, display_name=name,
                actor_id=admin["id"], actor_email=admin["email"], ip=self.client_ip(request),
            )
        except security.ValidationError as exc:
            return self.redirect(
                request, f"/?new=1&error={i18n.error_message(lang, str(exc))}"
            )
        except vs.DuplicateEmail:
            return self.redirect(
                request, f"/?new=1&error={i18n.error_message(lang, 'duplicate_email')}"
            )
        return self.redirect(request, "/")

    # -- user detail ----------------------------------------------------

    @login_required
    async def user_detail(self, request: Request) -> Response:
        user = self.store.get_user(int(request.path_params["user_id"]))
        if user is None:
            return self.redirect(request, "/")
        return self.render(
            request, "user.html", nav="dash",
            user=user, tokens=self.store.list_tokens(user["id"]),
            error=request.query_params.get("error", ""),
        )

    @login_required
    async def set_disabled(self, request: Request) -> Response:
        admin = self.current_admin(request)
        form = await self.form(request)
        user_id = int(request.path_params["user_id"])
        if not self.csrf_guard(request, form):
            return self.redirect(request, f"/users/{user_id}")
        disabled = request.url.path.endswith("/disable")
        try:
            self.store.set_user_disabled(
                user_id, disabled, actor_id=admin["id"], actor_email=admin["email"],
                ip=self.client_ip(request),
            )
        except vs.NotFound:
            return self.redirect(request, "/")
        return self.redirect(request, f"/users/{user_id}")

    # -- tokens ---------------------------------------------------------

    def _issue(self, make: Callable[[str], dict[str, Any]]) -> tuple[str, dict[str, Any]]:
        """Generate + persist, retrying on the (astronomical) hash collision."""
        last: Exception | None = None
        for _ in range(3):
            token = security.generate_token()
            try:
                return token, make(token)
            except vs.TokenCollision as exc:
                last = exc
        raise RuntimeError(f"could not issue a unique token: {last}")

    @login_required
    async def create_token(self, request: Request) -> Response:
        admin = self.current_admin(request)
        form = await self.form(request)
        lang = self.lang(request)
        user_id = int(request.path_params["user_id"])
        if not self.csrf_guard(request, form):
            return self.redirect(request, f"/users/{user_id}")

        try:
            label = security.normalize_label(form.get("label", ""))
        except security.ValidationError as exc:
            return self.redirect(
                request, f"/users/{user_id}?error={i18n.error_message(lang, str(exc))}"
            )

        try:
            plaintext, _ = self._issue(lambda tok: self.store.create_token(
                user_id=user_id, token=tok, label=label,
                actor_id=admin["id"], actor_email=admin["email"], ip=self.client_ip(request),
            ))
        except vs.NotFound:
            return self.redirect(request, "/")
        except vs.VaultError:
            return self.redirect(
                request, f"/users/{user_id}?error={i18n.error_message(lang, 'disabled_user')}"
            )

        user = self.store.get_user(user_id)
        return self._reveal(request, plaintext, f"{user['email']} — {label}", user_id)

    @login_required
    async def rotate_token(self, request: Request) -> Response:
        admin = self.current_admin(request)
        form = await self.form(request)
        token_id = int(request.path_params["token_id"])
        user_id = int(form.get("user_id") or 0)
        if not self.csrf_guard(request, form):
            return self.redirect(request, f"/users/{user_id}")

        try:
            plaintext, _ = self._issue(lambda tok: self.store.rotate_token(
                token_id, new_token=tok, grace_seconds=_grace_hours() * 3600,
                actor_id=admin["id"], actor_email=admin["email"], ip=self.client_ip(request),
            ))
        except (vs.NotFound, vs.SuccessorExists, vs.VaultError):
            # Already rotated (possibly by a second tab): show the list, which
            # now holds the successor. Never a 500.
            return self.redirect(request, f"/users/{user_id}")

        user = self.store.get_user(user_id)
        subject = f"{user['email']} — rotate" if user else "rotate"
        return self._reveal(request, plaintext, subject, user_id)

    def _reveal(
        self, request: Request, plaintext: str, subject: str, user_id: int
    ) -> Response:
        """The one and only response that carries a token in plaintext."""
        if request.headers.get("HX-Request"):
            return self.render(
                request, "partials/token_modal.html",
                token=plaintext, subject=subject, user_id=user_id,
            )
        # No-JS fallback: full page with the modal already open.
        user = self.store.get_user(user_id)
        return self.render(
            request, "user.html", nav="dash", user=user,
            tokens=self.store.list_tokens(user_id),
            token=plaintext, subject=subject, error="",
        )

    @login_required
    async def revoke_token(self, request: Request) -> Response:
        admin = self.current_admin(request)
        form = await self.form(request)
        token_id = int(request.path_params["token_id"])
        user_id = int(form.get("user_id") or 0)
        if self.csrf_guard(request, form):
            try:
                self.store.revoke_token(
                    token_id, actor_id=admin["id"], actor_email=admin["email"],
                    ip=self.client_ip(request),
                )
            except vs.NotFound:
                pass
        return self.redirect(request, f"/users/{user_id}")

    # -- audit ----------------------------------------------------------

    @login_required
    async def audit(self, request: Request) -> Response:
        try:
            offset = max(0, int(request.query_params.get("offset", "0")))
        except ValueError:
            offset = 0
        events = self.store.list_audit(limit=AUDIT_PAGE_SIZE, offset=offset)
        return self.render(
            request, "audit.html", nav="audit", events=events, offset=offset,
            page_size=AUDIT_PAGE_SIZE, total=self.store.count_audit(),
        )

    # -- ops ------------------------------------------------------------

    async def healthz(self, request: Request) -> Response:
        return JSONResponse({"status": "ok", "vault_db": vs.probe(self.store.db_path)})


# ---------------------------------------------------------------- factory


def create_app(db_path: str | None = None, *, secret_key: str | None = None) -> Starlette:
    key = secret_key or env("VAULT_SECRET_KEY")
    if not key:
        raise RuntimeError(
            "VAULT_SECRET_KEY is required (put it in .vault.env, mode 600) — "
            "without it sessions could be forged"
        )

    https_only = _https_only()
    store = vs.VaultStore(db_path or env("MEM0_VAULT_DB_PATH") or str(
        Path.home() / ".mem0" / "vault.db"
    ))
    r = Routes(store)

    routes = [
        Route("/login", r.login_page, methods=["GET"]),
        Route("/login", r.login_submit, methods=["POST"]),
        Route("/logout", r.logout, methods=["POST"]),
        Route("/", r.dashboard, methods=["GET"]),
        Route("/users", r.create_user, methods=["POST"]),
        Route("/users/{user_id:int}", r.user_detail, methods=["GET"]),
        Route("/users/{user_id:int}/disable", r.set_disabled, methods=["POST"]),
        Route("/users/{user_id:int}/enable", r.set_disabled, methods=["POST"]),
        Route("/users/{user_id:int}/tokens", r.create_token, methods=["POST"]),
        Route("/tokens/{token_id:int}/rotate", r.rotate_token, methods=["POST"]),
        Route("/tokens/{token_id:int}/revoke", r.revoke_token, methods=["POST"]),
        Route("/audit", r.audit, methods=["GET"]),
        Route("/healthz", r.healthz, methods=["GET"]),
        Mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static"),
    ]

    app = Starlette(
        routes=routes,
        middleware=[
            Middleware(
                SessionMiddleware, secret_key=key,
                session_cookie=_session_cookie_name(https_only),
                max_age=SESSION_MAX_AGE, same_site="lax", https_only=https_only,
            )
        ],
    )
    app.state.store = store
    return app
