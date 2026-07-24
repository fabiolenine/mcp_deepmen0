"""UI strings, PT and EN (the mockup's ``L()`` dictionary, server-side).

The toggle is a cookie read per request, so a page render never mixes
languages. Templates get the active dict as ``t``.
"""

from __future__ import annotations

LANGS = ("pt", "en")
DEFAULT_LANG = "pt"
LANG_COOKIE = "vault_lang"

PT = {
    "loginTitle": "Entrar no cofre",
    "loginSub": "Gestão de credenciais do DeepMem0 MCP",
    "email": "E-mail",
    "password": "Senha",
    "signIn": "Entrar",
    "lanNote": "Acesso restrito à LAN · sessão de 12 h",
    "navDash": "Visão geral",
    "navAudit": "Auditoria",
    "logout": "Sair",
    "adminRole": "Administrador",
    "dashSub": "Usuários e tokens que protegem o MCP na porta 8081",
    "statMode": "Modo de auth",
    "statUsers": "Usuários ativos",
    "statTokens": "Tokens ativos",
    "statUsed24h": "Tokens vistos · 24 h",
    "mcpLabel": "MCP · :8081",
    "mcpDown": "sem resposta",
    "ofTotal": "de",
    "newUser": "Novo usuário",
    "create": "Criar",
    "nuNamePh": "Nome de exibição",
    "nuEmailPh": "email@dominio.com",
    "nuNote": "Sem auto-registro — apenas admin cria usuários. E-mail validado por formato.",
    "colUser": "Usuário",
    "colTokens": "Tokens",
    "colStatus": "Status",
    "colLast": "Último uso",
    "colLabel": "Rótulo",
    "colPrefix": "Prefixo",
    "colCreated": "Criado",
    "colExpires": "Expira",
    "colActions": "Ações",
    "colWhen": "Quando",
    "colActor": "Autor",
    "colAction": "Ação",
    "colSubject": "Alvo",
    "active": "Ativo",
    "disabled": "Desativado",
    "revoked": "Revogado",
    "expiring": "Expirando",
    "never": "Nunca",
    "back": "Voltar",
    "disable": "Desativar usuário",
    "enable": "Reativar usuário",
    "genToken": "Gerar token",
    "rotate": "Renovar",
    "revoke": "Revogar",
    "rotateNote": (
        "Renovar emite um token novo; o antigo permanece válido por {hours} h "
        "(janela de migração). Revogar mata na hora."
    ),
    "revokeNote": (
        "A revogação vale para novas requisições — e toda chamada de tool do MCP "
        "é uma requisição nova."
    ),
    "auditNote": "Registro append-only — tokens nunca aparecem em claro, apenas o prefixo.",
    "modalTitle": "Token gerado",
    "modalWarn": (
        "Este token é exibido apenas uma vez. Copie agora e guarde em local seguro — "
        "depois desta tela, só o prefixo fica visível."
    ),
    "copy": "Copiar",
    "copied": "Copiado",
    "copyManual": "Selecionado — use Ctrl+C",
    "done": "Já copiei, fechar",
    "modeHintOff": "Auth desligada — tokens não são exigidos",
    "modeHintShadow": "Valida e registra, sem bloquear",
    "modeHintOn": "Requisições sem token = 401",
    "tokenLabelPh": "rótulo do token (ex: claude-code-local)",
    "noTokens": "Nenhum token emitido.",
    "noUsers": "Nenhum usuário ainda.",
    "noAudit": "Sem eventos registrados.",
    "prev": "Anteriores",
    "next": "Próximos",
    "err_email_required": "Informe um e-mail.",
    "err_email_invalid": "E-mail inválido — verifique o formato.",
    "err_name_required": "Informe um nome de exibição.",
    "err_name_too_long": "Nome muito longo.",
    "err_label_invalid": "Rótulo inválido — use letras, números, espaço, . @ : + -",
    "err_label_too_long": "Rótulo muito longo (máx. 60).",
    "err_duplicate_email": "Este e-mail já está cadastrado.",
    "err_bad_credentials": "E-mail ou senha incorretos.",
    "err_csrf": "Sessão expirada — recarregue a página e tente de novo.",
    "err_disabled_user": "Usuário desativado — reative antes de emitir tokens.",
    "modeUnconfirmed": "não confirmado — o :8081 não respondeu",
    "readyTitle": "Prontidão para MEM0_REQUIRE_AUTH=on",
    "readyYes": "Pronto para ligar",
    "readyNo": "Ainda não",
    "readySeen": "tokens ativos vistos na janela",
    "readySilent": "sem uso na janela (tomariam 401)",
    "readyDenials": "requisições negadas na janela",
    "readyNote": "Silêncio não é evidência: token que ninguém usou é indistinguível de cliente que quebraria.",
    "readyWindow": "janela",
    "noAdminTitle": "Nenhum administrador cadastrado",
    "noAdminBody": (
        "Rode <code>deepmem0-vault bootstrap-admin</code> no servidor para criar "
        "o primeiro administrador."
    ),
}

EN = {
    "loginTitle": "Sign in to the vault",
    "loginSub": "Credential management for the DeepMem0 MCP",
    "email": "Email",
    "password": "Password",
    "signIn": "Sign in",
    "lanNote": "LAN-only access · 12 h session",
    "navDash": "Overview",
    "navAudit": "Audit log",
    "logout": "Sign out",
    "adminRole": "Administrator",
    "dashSub": "Users and tokens protecting the MCP on port 8081",
    "statMode": "Auth mode",
    "statUsers": "Active users",
    "statTokens": "Active tokens",
    "statUsed24h": "Tokens seen · 24 h",
    "mcpLabel": "MCP · :8081",
    "mcpDown": "no answer",
    "ofTotal": "of",
    "newUser": "New user",
    "create": "Create",
    "nuNamePh": "Display name",
    "nuEmailPh": "email@domain.com",
    "nuNote": "No self-registration — only admins create users. Email is format-validated.",
    "colUser": "User",
    "colTokens": "Tokens",
    "colStatus": "Status",
    "colLast": "Last used",
    "colLabel": "Label",
    "colPrefix": "Prefix",
    "colCreated": "Created",
    "colExpires": "Expires",
    "colActions": "Actions",
    "colWhen": "When",
    "colActor": "Actor",
    "colAction": "Action",
    "colSubject": "Subject",
    "active": "Active",
    "disabled": "Disabled",
    "revoked": "Revoked",
    "expiring": "Expiring",
    "never": "Never",
    "back": "Back",
    "disable": "Disable user",
    "enable": "Re-enable user",
    "genToken": "Generate token",
    "rotate": "Rotate",
    "revoke": "Revoke",
    "rotateNote": (
        "Rotate issues a new token; the old one stays valid for {hours} h "
        "(migration window). Revoke kills it instantly."
    ),
    "revokeNote": (
        "Revocation applies to new requests — and every MCP tool call is a new request."
    ),
    "auditNote": "Append-only log — tokens never appear in plaintext, only the prefix.",
    "modalTitle": "Token generated",
    "modalWarn": (
        "This token is shown only once. Copy it now and store it safely — after this "
        "screen, only the prefix remains visible."
    ),
    "copy": "Copy",
    "copied": "Copied",
    "copyManual": "Selected — press Ctrl+C",
    "done": "Copied it, close",
    "modeHintOff": "Auth off — tokens not required",
    "modeHintShadow": "Validates and logs, never blocks",
    "modeHintOn": "Requests without a token = 401",
    "tokenLabelPh": "token label (e.g. claude-code-local)",
    "noTokens": "No tokens issued.",
    "noUsers": "No users yet.",
    "noAudit": "No events recorded.",
    "prev": "Previous",
    "next": "Next",
    "err_email_required": "Enter an email.",
    "err_email_invalid": "Invalid email — check the format.",
    "err_name_required": "Enter a display name.",
    "err_name_too_long": "Name is too long.",
    "err_label_invalid": "Invalid label — use letters, numbers, space, . @ : + -",
    "err_label_too_long": "Label is too long (max 60).",
    "err_duplicate_email": "That email is already registered.",
    "err_bad_credentials": "Wrong email or password.",
    "err_csrf": "Session expired — reload the page and try again.",
    "err_disabled_user": "User is disabled — re-enable before issuing tokens.",
    "modeUnconfirmed": "unconfirmed — :8081 did not answer",
    "readyTitle": "Readiness for MEM0_REQUIRE_AUTH=on",
    "readyYes": "Ready to flip",
    "readyNo": "Not yet",
    "readySeen": "active tokens seen in the window",
    "readySilent": "unused in the window (would take 401)",
    "readyDenials": "denied requests in the window",
    "readyNote": "Silence is not evidence: a token nobody used is indistinguishable from a client that would break.",
    "readyWindow": "window",
    "noAdminTitle": "No administrator registered",
    "noAdminBody": (
        "Run <code>deepmem0-vault bootstrap-admin</code> on the server to create "
        "the first administrator."
    ),
}

class Strings:
    """Template-facing string table.

    A plain dict is the wrong thing to hand Jinja: ``{{ t.copy }}`` resolves
    ``dict.copy`` (the METHOD) before the key, and the page renders
    "<built-in method copy of dict object...>". Every key that happens to
    share a name with a dict method — copy, keys, values, items, get, update,
    pop, clear — is a landmine. This exposes only the strings, so attribute
    and subscript access both mean "look up the key".
    """

    __slots__ = ("_table",)

    def __init__(self, table: dict[str, str]):
        object.__setattr__(self, "_table", table)

    def __getattr__(self, key: str) -> str:
        try:
            return self._table[key]
        except KeyError as exc:
            raise AttributeError(key) from exc

    def __getitem__(self, key: str) -> str:
        return self._table[key]

    def __contains__(self, key: str) -> bool:
        return key in self._table

    def __iter__(self):
        return iter(self._table)

    def keys(self):  # noqa: D102 - mapping protocol for tests/debugging
        return self._table.keys()


_TABLES = {"pt": Strings(PT), "en": Strings(EN)}


def normalize_lang(raw: str | None) -> str:
    lang = (raw or "").strip().lower()
    return lang if lang in LANGS else DEFAULT_LANG


def strings(lang: str) -> Strings:
    return _TABLES.get(normalize_lang(lang), _TABLES["pt"])


def error_message(lang: str, code: str) -> str:
    """Map a validation code to a localized message (unknown → the code)."""
    key = f"err_{code}"
    table = strings(lang)
    return table[key] if key in table else code
